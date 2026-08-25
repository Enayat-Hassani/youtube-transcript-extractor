"""ytx command-line interface."""

from __future__ import annotations

import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from ytx_core import (
    EXPORT_FORMATS,
    TranscriptService,
    YtxError,
    extract_video_id,
    format_transcript,
)
from ytx_core.cleanup import CleanupOptions, available_packs, clean
from ytx_core.doc import compose_index, compose_markdown_doc, fetch_video_metadata
from ytx_core.screen import extract_screen_text

__version__ = "0.1.0"

DEFAULT_DB = Path("./ytx_cache.sqlite3")

Format = StrEnum("Format", {value.upper(): value for value in EXPORT_FORMATS})

DbOption = Annotated[
    Path,
    typer.Option("--db", envvar="YTX_DB_PATH", help="SQLite cache database path."),
]

app = typer.Typer(name="ytx", help="ytx — YouTube transcript extractor", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ytx {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """YouTube transcript extractor."""


def _fail(message: str) -> NoReturn:
    typer.secho(f"[err] {message}", fg="red", err=True)
    raise typer.Exit(1)


def _build_service(db: Path, no_cache: bool) -> TranscriptService:
    return TranscriptService(db_path=db, enable_cache=not no_cache)


def _parse_languages(raw: Sequence[str]) -> list[str] | None:
    languages: list[str] = []
    for chunk in raw:
        languages.extend(part.strip() for part in chunk.split(",") if part.strip())
    return languages or None


def _fetch(
    service: TranscriptService,
    url: str,
    languages: list[str] | None,
    refresh: bool,
    fmt: str,
    translate_to: str | None = None,
):
    started = time.perf_counter()
    document = service.get(url, languages=languages, refresh=refresh, translate_to=translate_to)
    rendered = format_transcript(document, fmt)
    elapsed = time.perf_counter() - started
    return document, rendered, elapsed


@app.command()
def get(
    url: Annotated[str, typer.Argument(help="YouTube URL or video ID.")],
    lang: Annotated[
        list[str],
        typer.Option("--lang", "-l", help='Preferred languages, e.g. "en,de" (repeatable).'),
    ] = [],
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.JSON,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
    translate: Annotated[
        str | None,
        typer.Option(
            "--translate", "-t", help="Translate captions to this language via YouTube (e.g. en)."
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Bypass cached results.")] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable cache reads and writes for this run."),
    ] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Fetch a transcript and print or save it in the chosen format."""
    service = _build_service(db, no_cache)
    try:
        document, rendered, elapsed = _fetch(
            service, url, _parse_languages(lang), refresh, fmt.value, translate
        )
    except YtxError as exc:
        _fail(str(exc))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        typer.echo(rendered)
    typer.secho(
        f"[ok] {document.video_id} · {document.language} · {len(document.segments)} segments"
        f" · {document.source.backend}/{document.source.kind.value} · {elapsed:.2f}s",
        fg="green",
        err=True,
    )


def _parse_domains(values: list[str]) -> tuple[str, ...] | None:
    """`--domain` values to explicit packs; None means auto-detect."""
    names = [item.strip() for value in values for item in value.split(",") if item.strip()]
    if not names:
        return None
    if any(name.lower() == "none" for name in names):
        return ()
    return tuple(names)


@app.command()
def doc(
    url: Annotated[str, typer.Argument(help="YouTube URL or video ID.")],
    lang: Annotated[
        list[str],
        typer.Option("--lang", "-l", help='Preferred languages, e.g. "en,de" (repeatable).'),
    ] = [],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
    translate: Annotated[
        str | None,
        typer.Option(
            "--translate", "-t", help="Translate captions to this language via YouTube (e.g. en)."
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Bypass cached results.")] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable cache reads and writes for this run."),
    ] = False,
    no_clean: Annotated[
        bool,
        typer.Option("--no-clean", help="Emit the raw transcript with no cleanup passes."),
    ] = False,
    keep_sponsors: Annotated[
        bool,
        typer.Option("--keep-sponsors", help="Keep sponsor and self-promo blocks."),
    ] = False,
    no_sponsorblock: Annotated[
        bool,
        typer.Option("--no-sponsorblock", help="Do not query the SponsorBlock database."),
    ] = False,
    no_fix_terms: Annotated[
        bool,
        typer.Option("--no-fix-terms", help="Skip vocabulary repair."),
    ] = False,
    domain: Annotated[
        list[str],
        typer.Option(
            "--domain",
            "-d",
            help='Lexicon packs, e.g. "trading,tech".'
            ' Omit to auto-detect; "none" for general only.',
        ),
    ] = [],
    lexicon: Annotated[
        Path | None,
        typer.Option("--lexicon", help="Path to a custom TOML lexicon pack."),
    ] = None,
    index: Annotated[
        bool,
        typer.Option(
            "--index",
            help="Write the document and print a compact map of it instead of its text.",
        ),
    ] = False,
    frames: Annotated[
        bool,
        typer.Option(
            "--frames",
            help="OCR on-screen text (charts, dashboards). Needs ffmpeg + tesseract.",
        ),
    ] = False,
    frame_interval: Annotated[
        float,
        typer.Option("--frame-interval", help="Seconds between sampled frames."),
    ] = 120.0,
    max_frames: Annotated[
        int,
        typer.Option("--max-frames", help="Hard cap on frames sampled per video."),
    ] = 40,
    keep_frames: Annotated[
        Path | None,
        typer.Option(
            "--keep-frames",
            help="Also save the sampled frames here. Off by default: frames are temporary.",
        ),
    ] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Fetch a transcript and render it as an AI-ready Markdown document."""
    service = _build_service(db, no_cache)
    try:
        document = service.get(
            url, languages=_parse_languages(lang), refresh=refresh, translate_to=translate
        )
        video_id = extract_video_id(url)
        metadata = fetch_video_metadata(video_id)
        options = (
            CleanupOptions.disabled()
            if no_clean
            else CleanupOptions(
                fix_terms=not no_fix_terms,
                packs=_parse_domains(domain),
                custom_pack=str(lexicon) if lexicon else None,
                strip_sponsors=not keep_sponsors,
                use_sponsorblock=not no_sponsorblock,
                clean_description=True,
            )
        )
        document, description, report = clean(
            document,
            video_id=video_id,
            title=metadata.title,
            description=metadata.description,
            options=options,
        )
        if description != metadata.description:
            metadata = metadata.model_copy(update={"description": description})
        captures = []
        if frames:
            typer.secho("[..] sampling frames for on-screen text", fg="cyan", err=True)
            screen = extract_screen_text(
                video_id,
                metadata.duration_sec or document.last_end,
                interval=frame_interval,
                max_frames=max_frames,
                keep_frames=keep_frames,
            )
            captures = screen.captures
            if screen.status:
                typer.secho(f"[..] screen text: {screen.status}", fg="yellow", err=True)
        markdown = compose_markdown_doc(
            metadata, document, notes=report.notes, screen=captures
        )
    except YtxError as exc:
        _fail(str(exc))
    if index and output is None:
        output = Path(f"{video_id}.md")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    if index:
        typer.echo(
            compose_index(
                metadata, document, path=str(output), notes=report.notes, screen=captures
            ),
            nl=False,
        )
    elif output is None:
        typer.echo(markdown)
    typer.secho(f"[ok] doc {video_id} · {len(markdown)} chars", fg="green", err=True)


def _read_urls(path: Path) -> list[str]:
    if not path.is_file():
        _fail(f"input file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = (line.strip() for line in lines)
    return [line for line in stripped if line and not line.startswith("#")]


@app.command()
def batch(
    file: Annotated[
        Path,
        typer.Argument(help="Text file with one YouTube URL or video ID per line."),
    ],
    lang: Annotated[
        list[str],
        typer.Option("--lang", "-l", help='Preferred languages, e.g. "en,de" (repeatable).'),
    ] = [],
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.JSON,
    outdir: Annotated[
        Path,
        typer.Option("--outdir", help="Directory for transcript files."),
    ] = Path("transcripts"),
    translate: Annotated[
        str | None,
        typer.Option(
            "--translate", "-t", help="Translate captions to this language via YouTube (e.g. en)."
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Bypass cached results.")] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable cache reads and writes for this run."),
    ] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Fetch transcripts for every URL in FILE, writing one file per video."""
    urls = _read_urls(file)
    if not urls:
        typer.secho(f"No URLs found in {file}", fg="yellow")
        return
    service = _build_service(db, no_cache)
    outdir.mkdir(parents=True, exist_ok=True)
    languages = _parse_languages(lang)
    ok = 0
    failed = 0
    total = len(urls)
    for index, url in enumerate(urls, start=1):
        try:
            document, rendered, _ = _fetch(service, url, languages, refresh, fmt.value, translate)
        except YtxError as exc:
            failed += 1
            typer.secho(f"[{index}/{total}] ✗ {url} — {exc}", fg="red")
            continue
        target = outdir / f"{document.video_id}.{fmt.value}"
        target.write_text(rendered, encoding="utf-8")
        ok += 1
        typer.secho(
            f"[{index}/{total}] ✓ {document.video_id}"
            f" ({document.language}, {len(document.segments)} segs)",
            fg="green",
        )
    typer.echo(f"Done: {ok} ok, {failed} failed")
    if failed:
        raise typer.Exit(1)


@app.command()
def langs(
    url: Annotated[str, typer.Argument(help="YouTube URL or video ID.")],
    db: DbOption = DEFAULT_DB,
) -> None:
    """List available transcript languages for a video."""
    service = _build_service(db, no_cache=False)
    try:
        options = service.list_languages(url)
    except YtxError as exc:
        _fail(str(exc))
    typer.echo(f"Available transcripts for {url}:")
    table = Table()
    table.add_column("code")
    table.add_column("label")
    table.add_column("kind")
    table.add_column("translatable")
    for option in options:
        table.add_row(
            option.language_code,
            option.language_label or "",
            option.kind.value if option.kind else "",
            "yes" if option.is_translatable else "no",
        )
    Console().print(table)


@app.command()
def health(db: DbOption = DEFAULT_DB) -> None:
    """Show backend health from the circuit breaker."""
    service = _build_service(db, no_cache=False)
    try:
        backends = service.health()["backends"]
    except YtxError as exc:
        _fail(str(exc))
    table = Table()
    table.add_column("backend")
    table.add_column("state")
    table.add_column("consecutive_failures")
    for row in backends:
        table.add_row(
            str(row.get("backend", "")),
            str(row.get("state", "")),
            str(row.get("consecutive_failures", 0)),
        )
    Console().print(table)


@app.command()
def packs() -> None:
    """List the bundled lexicon packs used by `ytx doc`."""
    for name, pack in sorted(available_packs().items()):
        typer.echo(f"{name:<10} {len(pack.rules):>3} rules  {pack.description}")
