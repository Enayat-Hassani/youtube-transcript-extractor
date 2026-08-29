"""FastAPI service exposing ytx-core transcript extraction."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from ytx_core import (
    EXPORT_FORMATS,
    AllBackendsFailedError,
    InvalidInputError,
    NoTranscriptFoundError,
    PlaylistNotSupportedError,
    TranscriptDocument,
    TranscriptsDisabledError,
    TranscriptService,
    VideoUnavailableError,
    YtxError,
    format_transcript,
)
from ytx_core.cleanup import CleanupOptions, clean
from ytx_core.doc import compose_markdown_doc, fetch_video_metadata
from ytx_core.jobs import JobRunner
from ytx_core.screen import extract_screen_text

_MEDIA_TYPES = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "md": "text/plain",
}

ExportFormat = StrEnum("ExportFormat", {fmt.upper(): fmt for fmt in EXPORT_FORMATS})

MAX_BATCH_URLS = 500


class BatchRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    languages: list[str] | None = None
    refresh: bool = False

    @field_validator("urls")
    @classmethod
    def _limit_url_count(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_BATCH_URLS:
            raise ValueError(f"urls must contain at most {MAX_BATCH_URLS} items")
        return value


class NotFound(Exception):
    pass


def _package_version() -> str:
    try:
        return _metadata_version("ytx-api")
    except PackageNotFoundError:
        return "0.0.0"


def _service_from_env() -> TranscriptService:
    db_path = os.environ.get("YTX_DB_PATH") or None
    raw_disable = os.environ.get("YTX_DISABLE_CACHE", "").strip().lower()
    enable_cache = raw_disable not in {"1", "true"}
    return TranscriptService(db_path=db_path, enable_cache=enable_cache)


def _close_service(service: object) -> None:
    closer = getattr(service, "close", None)
    if callable(closer):
        closer()


def _shutdown_runner(runner: object) -> None:
    shutdown = getattr(runner, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=False)


def _parse_languages(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    languages = [part.strip() for part in raw.split(",")]
    filtered = [lang for lang in languages if lang]
    return filtered or None


def _compose_clean_doc(
    doc: TranscriptDocument,
    video_id: str,
    *,
    frames: bool,
    keep_sponsors: bool,
    use_sponsorblock: bool,
    fix_terms: bool,
) -> str:
    """Run the `ytx doc` pipeline (sponsor removal, vocabulary repair, and
    optional on-screen OCR) and return an AI-ready Markdown document."""
    metadata = fetch_video_metadata(video_id)
    options = CleanupOptions(
        fix_terms=fix_terms,
        strip_sponsors=not keep_sponsors,
        use_sponsorblock=use_sponsorblock,
        clean_description=True,
    )
    doc, description, report = clean(
        doc,
        video_id=video_id,
        title=metadata.title,
        description=metadata.description,
        options=options,
    )
    if description != metadata.description:
        metadata = metadata.model_copy(update={"description": description})
    captures = []
    if frames:
        # Needs ffmpeg + an OCR engine; degrades to no captures (with a status)
        # when they're missing, so this never hard-fails the request.
        screen = extract_screen_text(video_id, metadata.duration_sec or doc.last_end)
        captures = screen.captures
    return compose_markdown_doc(metadata, doc, notes=report.notes, screen=captures)


def _attachment_headers(video_id: str, fmt: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{video_id}.{fmt}"'}


def _error_body(exc: Exception, extra: dict[str, object] | None = None) -> dict[str, object]:
    error: dict[str, object] = {"type": type(exc).__name__, "message": str(exc)}
    if extra:
        error.update(extra)
    return {"error": error}


def _json_error(
    status_code: int, exc: Exception, extra: dict[str, object] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_error_body(exc, extra))


async def _handle_all_backends_failed(
    request: Request, exc: AllBackendsFailedError
) -> JSONResponse:
    attempts = [
        {"backend": attempt.backend, "message": attempt.message, "retryable": attempt.retryable}
        for attempt in exc.attempts
    ]
    return _json_error(502, exc, {"attempts": attempts})


async def _handle_ytx_error(request: Request, exc: YtxError) -> JSONResponse:
    return _json_error(400, exc)


async def _handle_not_found(request: Request, exc: YtxError) -> JSONResponse:
    return _json_error(404, exc)


def create_app(
    service: TranscriptService | None = None, runner: JobRunner | None = None
) -> FastAPI:
    """Build the ytx-api FastAPI application around an optional service and job runner."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            app.state.service = _service_from_env()
        if app.state.runner is None:
            app.state.runner = JobRunner(
                lambda: app.state.service,
                os.environ.get("YTX_JOBS_DB_PATH", "./ytx_jobs.sqlite3"),
                max_workers=int(os.environ.get("YTX_MAX_JOB_WORKERS", "4")),
            )
        yield
        _close_service(app.state.service)
        _shutdown_runner(app.state.runner)

    app = FastAPI(title="ytx-api", version=_package_version(), lifespan=lifespan)
    app.state.service = service
    app.state.runner = runner

    @app.get("/api/v1/transcripts/{video_id}", response_model=TranscriptDocument)
    def get_transcript(
        request: Request,
        video_id: str,
        languages: str | None = None,
        format: ExportFormat = ExportFormat.JSON,
        download: bool = False,
        refresh: bool = False,
        translate: str | None = None,
        clean: bool = False,
        frames: bool = False,
        keep_sponsors: bool = False,
        sponsorblock: bool = True,
        fix_terms: bool = True,
    ) -> object:
        svc: TranscriptService = request.app.state.service
        doc = svc.get(
            video_id,
            languages=_parse_languages(languages),
            refresh=refresh,
            translate_to=translate,
        )

        # `clean`/`frames` run the full doc pipeline and always return Markdown,
        # regardless of `format` (the composed document is inherently Markdown).
        if clean or frames:
            markdown = _compose_clean_doc(
                doc,
                video_id,
                frames=frames,
                keep_sponsors=keep_sponsors,
                use_sponsorblock=sponsorblock,
                fix_terms=fix_terms,
            )
            headers = _attachment_headers(video_id, "md") if download else None
            return PlainTextResponse(
                markdown, media_type="text/markdown; charset=utf-8", headers=headers
            )

        fmt = format.value
        headers = _attachment_headers(video_id, fmt) if download else None
        if format is ExportFormat.JSON:
            if headers:
                return JSONResponse(content=doc.model_dump(mode="json"), headers=headers)
            return doc
        media_type = f"{_MEDIA_TYPES.get(fmt, 'text/plain')}; charset=utf-8"
        return PlainTextResponse(
            format_transcript(doc, fmt),
            media_type=media_type,
            headers=headers,
        )

    @app.get("/api/v1/videos/{video_id}")
    def list_video_languages(request: Request, video_id: str) -> dict[str, object]:
        svc: TranscriptService = request.app.state.service
        options = svc.list_languages(video_id)
        return {"video_id": video_id, "languages": options}

    @app.post("/api/v1/transcripts", status_code=202)
    def submit_batch(request: Request, payload: BatchRequest) -> dict[str, object]:
        runner: JobRunner = request.app.state.runner
        job_id = runner.submit(payload.urls, languages=payload.languages, refresh=payload.refresh)
        return {"job_id": job_id, "count": len(payload.urls)}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> object:
        runner: JobRunner = request.app.state.runner
        job = runner.get_job(job_id)
        if job is None:
            return _json_error(404, NotFound("job not found"))
        return job

    @app.get("/api/v1/jobs")
    def list_jobs(
        request: Request,
        limit: Annotated[int, Query(le=100)] = 20,
    ) -> dict[str, object]:
        runner: JobRunner = request.app.state.runner
        return {"jobs": runner.list_jobs(limit)}

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        svc: TranscriptService = request.app.state.service
        return {
            "status": "ok",
            "version": _package_version(),
            "backends": svc.health().get("backends", []),
        }

    app.add_exception_handler(AllBackendsFailedError, _handle_all_backends_failed)
    app.add_exception_handler(PlaylistNotSupportedError, _handle_ytx_error)
    app.add_exception_handler(InvalidInputError, _handle_ytx_error)
    app.add_exception_handler(VideoUnavailableError, _handle_not_found)
    app.add_exception_handler(TranscriptsDisabledError, _handle_not_found)
    app.add_exception_handler(NoTranscriptFoundError, _handle_not_found)
    app.add_exception_handler(YtxError, _handle_ytx_error)
    return app


app = create_app()
