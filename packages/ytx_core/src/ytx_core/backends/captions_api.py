from __future__ import annotations

from typing import Any

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.errors import (
    BackendError,
    NoTranscriptFoundError,
    TranscriptsDisabledError,
    VideoUnavailableError,
    YtxError,
)
from ytx_core.models import LanguageOption, Segment, SourceInfo, SourceKind, TranscriptDocument


class CaptionsApiBackend(TranscriptBackend):
    """Backend backed by the youtube-transcript-api 1.x instance API."""

    name = "captions_api"
    supports_language_listing = True

    def _api(self) -> YouTubeTranscriptApi:
        return YouTubeTranscriptApi()

    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        requested = list(request.languages) if request.languages else []
        try:
            listing = list(self._api().list(request.video_id))
            chosen = self._select(listing, request)
            data = self._fetch_data(chosen, listing, request)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
            self._raise_translated(request.video_id, requested, exc)
        except YtxError:
            raise
        except Exception as exc:
            raise BackendError(self.name, f"{type(exc).__name__}: {exc}") from exc
        segments = [Segment(start=s.start, end=s.start + s.duration, text=s.text) for s in data]
        return TranscriptDocument(
            video_id=request.video_id,
            language=data.language_code,
            language_label=data.language,
            is_generated=data.is_generated,
            duration_sec=segments[-1].end if segments else 0.0,
            segments=segments,
            source=SourceInfo(
                kind=SourceKind.AUTO_CAPTIONS if data.is_generated else SourceKind.MANUAL_CAPTIONS,
                backend=self.name,
            ),
        )

    def list_transcripts(self, video_id: str) -> list[LanguageOption]:
        try:
            listing = list(self._api().list(video_id))
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
            self._raise_translated(video_id, [], exc)
        except YtxError:
            raise
        except Exception as exc:
            raise BackendError(self.name, f"{type(exc).__name__}: {exc}") from exc
        return [
            LanguageOption(
                language_code=t.language_code,
                language_label=t.language,
                kind=SourceKind.AUTO_CAPTIONS if t.is_generated else SourceKind.MANUAL_CAPTIONS,
                is_translatable=t.is_translatable,
            )
            for t in listing
        ]

    def _fetch_data(self, chosen: Any, listing: list[Any], request: FetchRequest) -> Any:
        target = request.translate_to
        if not target or self._base(chosen.language_code) == self._base(target):
            return chosen.fetch()
        candidate = chosen if chosen.is_translatable else None
        if candidate is None:
            candidate = next(
                (
                    t
                    for t in listing
                    if t.is_translatable
                    and self._base(t.language_code) == self._base(chosen.language_code)
                ),
                None,
            )
        if candidate is None:
            return chosen.fetch()
        return candidate.translate(self._base(target)).fetch()

    @staticmethod
    def _base(code: str) -> str:
        return code.split("-", 1)[0].lower()

    @staticmethod
    def _matches(code: str, want: str) -> bool:
        base = want.split("-", 1)[0].lower()
        lowered = code.lower()
        return lowered == base or lowered.startswith(f"{base}-")

    def _select(self, listing: list[Any], request: FetchRequest) -> Any:
        if request.languages:
            for code in request.languages:
                for transcript in listing:
                    if self._matches(transcript.language_code, code):
                        return transcript
            raise NoTranscriptFoundError(request.video_id, requested=list(request.languages))
        for transcript in listing:
            if not transcript.is_generated and self._matches(transcript.language_code, "en"):
                return transcript
        for transcript in listing:
            if not transcript.is_generated:
                return transcript
        return listing[0]

    def _raise_translated(self, video_id: str, requested: list[str], exc: Exception) -> None:
        if isinstance(exc, TranscriptsDisabled):
            raise TranscriptsDisabledError() from exc
        if isinstance(exc, NoTranscriptFound):
            raise NoTranscriptFoundError(video_id, requested=requested) from exc
        raise VideoUnavailableError(str(exc)) from exc
