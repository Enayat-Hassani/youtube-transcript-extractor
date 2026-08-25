from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ytx_core.backends import FetchRequest, TranscriptBackend, default_backends
from ytx_core.cache import TranscriptCache
from ytx_core.cascade import Cascade
from ytx_core.errors import (
    AllBackendsFailedError,
    AttemptRecord,
    BackendError,
    InvalidInputError,
    NoTranscriptFoundError,
    PlaylistNotSupportedError,
    TranscriptsDisabledError,
    VideoUnavailableError,
)
from ytx_core.expansion import expand_to_video_ids
from ytx_core.models import LanguageOption, SourceKind, TranscriptDocument
from ytx_core.resolver import extract_video_id

AUTO_TTL_SECONDS = 30 * 86400

_DEFINITIVE_ERRORS = (
    InvalidInputError,
    PlaylistNotSupportedError,
    VideoUnavailableError,
    TranscriptsDisabledError,
    NoTranscriptFoundError,
)


class TranscriptService:
    """Sync facade: resolve -> cascade fetch -> optional SQLite cache."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        enable_cache: bool = True,
        backends: Sequence[TranscriptBackend] | None = None,
        failure_threshold: int = 5,
        reset_timeout_s: float = 60.0,
    ) -> None:
        resolved_db_path = db_path or os.environ.get("YTX_DB_PATH") or "./ytx_cache.sqlite3"
        self.backends: list[TranscriptBackend] = (
            list(backends) if backends is not None else default_backends()
        )
        self.cascade = Cascade(
            self.backends,
            failure_threshold=failure_threshold,
            reset_timeout_s=reset_timeout_s,
        )
        self._cache = TranscriptCache(resolved_db_path) if enable_cache else None

    def get(
        self,
        url_or_id: str,
        *,
        languages: Sequence[str] | None = None,
        refresh: bool = False,
        translate_to: str | None = None,
    ) -> TranscriptDocument:
        video_id = extract_video_id(url_or_id)
        langs = tuple(languages) if languages else None
        key_lang = (langs[0] if langs else "") + (f">{translate_to}" if translate_to else "")
        if self._cache is not None and not refresh:
            hit = self._cache.get(video_id, key_lang)
            if hit is not None:
                return hit
        doc = self.cascade.fetch(FetchRequest(video_id, langs, translate_to=translate_to))
        if self._cache is not None:
            ttl = AUTO_TTL_SECONDS if doc.source.kind != SourceKind.MANUAL_CAPTIONS else None
            self._cache.put(doc, ttl_seconds=ttl, language=key_lang)
        return doc

    def list_languages(self, url_or_id: str) -> list[LanguageOption]:
        video_id = extract_video_id(url_or_id)
        attempts: list[AttemptRecord] = []
        for backend in self.backends:
            if not backend.supports_language_listing:
                continue
            try:
                return backend.list_transcripts(video_id)
            except _DEFINITIVE_ERRORS as exc:
                attempts.append(AttemptRecord(backend=backend.name, message=str(exc)))
                raise
            except BackendError as exc:
                attempts.append(
                    AttemptRecord(backend=backend.name, message=str(exc), retryable=True)
                )
        raise AllBackendsFailedError(attempts)

    def expand(self, url_or_id: str, *, limit: int = 500) -> list[str]:
        """Expand any input (video, playlist, channel, @handle) to video ids."""
        try:
            return [extract_video_id(url_or_id)]
        except InvalidInputError:
            pass
        return expand_to_video_ids(url_or_id, limit=limit)

    def health(self) -> dict:
        return {"backends": self.cascade.health()}

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
