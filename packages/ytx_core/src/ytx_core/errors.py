from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel


class YtxError(Exception):
    """Base class for all user-facing ytx errors."""


class InvalidInputError(YtxError):
    """URL or video ID could not be parsed."""


class PlaylistNotSupportedError(InvalidInputError):
    """A playlist URL was given; playlist expansion arrives in Phase 2."""


class LexiconPackError(InvalidInputError):
    """A lexicon pack is unknown, missing, or malformed."""


class VideoUnavailableError(YtxError):
    """Video is private, deleted, age-restricted, or otherwise unplayable."""


class TranscriptsDisabledError(YtxError):
    """The creator has disabled captions for this video."""


class NoTranscriptFoundError(YtxError):
    """No transcript exists for the requested languages (or at all)."""

    def __init__(self, video_id: str, requested: Sequence[str] | None = None) -> None:
        self.video_id = video_id
        self.requested = list(requested) if requested else None
        want = ", ".join(self.requested) if self.requested else "any"
        super().__init__(f"No transcript found for {video_id} (requested: {want})")


class BackendError(YtxError):
    """A single backend failed; retryable failures count toward its breaker."""

    def __init__(self, backend: str, message: str, *, retryable: bool = True) -> None:
        self.backend = backend
        self.retryable = retryable
        super().__init__(f"[{backend}] {message}")


class AttemptRecord(BaseModel):
    backend: str
    ok: bool = False
    message: str
    retryable: bool = False


class AllBackendsFailedError(YtxError):
    """Every enabled backend failed. Carries the per-backend attempt log."""

    def __init__(self, attempts: Sequence[AttemptRecord]) -> None:
        self.attempts = list(attempts)
        detail = "; ".join(f"{a.backend}: {a.message}" for a in self.attempts)
        detail = detail or "no backends attempted"
        super().__init__(f"All backends failed ({detail})")
