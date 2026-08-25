from __future__ import annotations

from collections.abc import Sequence

from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.breaker import CircuitBreaker
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
from ytx_core.models import TranscriptDocument

_DEFINITIVE_ERRORS = (
    InvalidInputError,
    PlaylistNotSupportedError,
    VideoUnavailableError,
    TranscriptsDisabledError,
    NoTranscriptFoundError,
)


class Cascade:
    """Try backends in order, guarded by a per-backend circuit breaker."""

    def __init__(
        self,
        backends: Sequence[TranscriptBackend],
        *,
        failure_threshold: int = 5,
        reset_timeout_s: float = 60.0,
    ) -> None:
        self.backends = list(backends)
        self.breakers = {
            backend.name: CircuitBreaker(
                backend.name,
                failure_threshold=failure_threshold,
                reset_timeout_s=reset_timeout_s,
            )
            for backend in self.backends
        }

    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        attempts: list[AttemptRecord] = []
        for backend in self.backends:
            breaker = self.breakers[backend.name]
            if not breaker.allow():
                continue
            try:
                doc = backend.fetch(request)
            except _DEFINITIVE_ERRORS as exc:
                attempts.append(
                    AttemptRecord(backend=backend.name, ok=False, message=str(exc), retryable=False)
                )
            except BackendError as exc:
                attempts.append(
                    AttemptRecord(
                        backend=backend.name,
                        ok=False,
                        message=str(exc),
                        retryable=exc.retryable,
                    )
                )
                breaker.record_failure(str(exc))
            else:
                breaker.record_success()
                return doc
        raise AllBackendsFailedError(attempts)

    def health(self) -> list[dict]:
        return [breaker.snapshot() for breaker in self.breakers.values()]
