from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker guarding a single backend."""

    def __init__(
        self, name: str, failure_threshold: int = 5, reset_timeout_s: float = 60.0
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._lock = threading.Lock()
        self._state = _CLOSED
        self._consecutive_failures = 0
        self._opened_at_mono: float | None = None
        self._opened_at_wall: datetime | None = None
        self._last_error: str | None = None
        self._probe_in_flight = False

    def allow(self) -> bool:
        """Return True when a call may go through (half-open allows one probe)."""
        with self._lock:
            if self._state == _OPEN:
                assert self._opened_at_mono is not None
                if time.monotonic() - self._opened_at_mono >= self.reset_timeout_s:
                    self._state = _HALF_OPEN
                    self._probe_in_flight = False
                else:
                    return False
            if self._state == _HALF_OPEN:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = _CLOSED
            self._consecutive_failures = 0
            self._opened_at_mono = None
            self._opened_at_wall = None
            self._last_error = None
            self._probe_in_flight = False

    def record_failure(self, error: str | Exception | None = None) -> None:
        with self._lock:
            self._last_error = (
                error
                if isinstance(error, str)
                else (str(error) if error is not None else self._last_error)
            )
            if self._state == _HALF_OPEN:
                self._trip()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = _OPEN
        self._opened_at_mono = time.monotonic()
        self._opened_at_wall = datetime.now(UTC)
        self._probe_in_flight = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "backend": self.name,
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "opened_at": (self._opened_at_wall.isoformat() if self._opened_at_wall else None),
                "last_error": self._last_error,
            }
