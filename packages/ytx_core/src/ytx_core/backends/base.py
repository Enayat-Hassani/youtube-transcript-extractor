from __future__ import annotations

from abc import ABC, abstractmethod

from ytx_core.models import LanguageOption, TranscriptDocument


class FetchRequest:
    __slots__ = ("video_id", "languages", "translate_to")

    def __init__(
        self,
        video_id: str,
        languages: tuple[str, ...] | None = None,
        translate_to: str | None = None,
    ) -> None:
        self.video_id = video_id
        self.languages = tuple(languages) if languages else None
        self.translate_to = translate_to


class TranscriptBackend(ABC):
    """A single source of transcripts.

    Backends return fully-formed TranscriptDocuments (including source info)
    or raise. Health/availability semantics:

    - Definitive negative answers raise ytx_core.errors subclasses
      (TranscriptsDisabledError, NoTranscriptFoundError, VideoUnavailableError).
      These do NOT count against the circuit breaker.
    - Anything transient or unexpected raises BackendError(retryable=True),
      which DOES count toward the circuit breaker.
    """

    name: str = "unnamed"
    supports_language_listing: bool = False

    @abstractmethod
    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        raise NotImplementedError

    def list_transcripts(self, video_id: str) -> list[LanguageOption]:
        raise NotImplementedError(f"backend {self.name} does not support listing")
