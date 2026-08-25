from __future__ import annotations

import threading

import pytest

from ytx_core.breaker import CircuitBreaker


class TestClosedState:
    def test_starts_closed_and_allowing(self):
        breaker = CircuitBreaker("b")
        assert breaker.allow() is True

    def test_failures_below_threshold_keep_calling(self):
        breaker = CircuitBreaker("b", failure_threshold=3)
        for _ in range(2):
            breaker.allow()
            breaker.record_failure(RuntimeError("boom"))
        assert breaker.allow() is True
        snapshot = breaker.snapshot()
        assert snapshot["state"] == "closed"
        assert snapshot["consecutive_failures"] == 2

    def test_threshold_trips_open(self):
        breaker = CircuitBreaker("b", failure_threshold=2)
        for _ in range(2):
            breaker.allow()
            breaker.record_failure(ValueError("boom"))
        assert breaker.allow() is False
        snapshot = breaker.snapshot()
        assert snapshot["state"] == "open"
        assert snapshot["last_error"] == "boom"

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker("b", failure_threshold=3)
        breaker.allow()
        breaker.record_failure("one")
        breaker.allow()
        breaker.record_failure("two")
        breaker.allow()
        breaker.record_success()
        snapshot = breaker.snapshot()
        assert snapshot["state"] == "closed"
        assert snapshot["consecutive_failures"] == 0
        assert snapshot["last_error"] is None


class TestOpenState:
    def test_open_blocks_until_reset_timeout(self, monkeypatch: pytest.MonkeyPatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("ytx_core.breaker.time.monotonic", lambda: clock["now"])
        breaker = CircuitBreaker("b", failure_threshold=1, reset_timeout_s=10.0)
        breaker.allow()
        breaker.record_failure("boom")
        clock["now"] = 9.9
        assert breaker.allow() is False
        clock["now"] = 10.0
        assert breaker.allow() is True

    def test_snapshot_records_opened_at(self):
        breaker = CircuitBreaker("b", failure_threshold=1)
        breaker.allow()
        breaker.record_failure(Exception("x"))
        snapshot = breaker.snapshot()
        assert snapshot["backend"] == "b"
        assert snapshot["opened_at"] is not None


class TestHalfOpenState:
    def test_half_open_admits_single_probe(self, monkeypatch: pytest.MonkeyPatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("ytx_core.breaker.time.monotonic", lambda: clock["now"])
        breaker = CircuitBreaker("b", failure_threshold=1, reset_timeout_s=5.0)
        breaker.allow()
        breaker.record_failure("boom")
        clock["now"] = 6.0
        assert breaker.allow() is True
        assert breaker.allow() is False

    def test_failed_probe_reopens_immediately(self, monkeypatch: pytest.MonkeyPatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("ytx_core.breaker.time.monotonic", lambda: clock["now"])
        breaker = CircuitBreaker("b", failure_threshold=1, reset_timeout_s=5.0)
        breaker.allow()
        breaker.record_failure("boom")
        clock["now"] = 6.0
        assert breaker.allow() is True
        breaker.record_failure("still broken")
        assert breaker.allow() is False
        assert breaker.snapshot()["state"] == "open"

    def test_successful_probe_closes(self, monkeypatch: pytest.MonkeyPatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("ytx_core.breaker.time.monotonic", lambda: clock["now"])
        breaker = CircuitBreaker("b", failure_threshold=1, reset_timeout_s=5.0)
        breaker.allow()
        breaker.record_failure("boom")
        clock["now"] = 6.0
        assert breaker.allow() is True
        breaker.record_success()
        assert breaker.snapshot()["state"] == "closed"
        clock["now"] = 100.0
        assert breaker.allow() is True


class TestSnapshot:
    def test_contains_expected_fields(self):
        breaker = CircuitBreaker("my-backend")
        snapshot = breaker.snapshot()
        assert set(snapshot) == {
            "backend",
            "state",
            "consecutive_failures",
            "opened_at",
            "last_error",
        }


class TestConcurrency:
    def test_thread_smoke(self):
        breaker = CircuitBreaker("b", failure_threshold=5)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    if breaker.allow():
                        breaker.record_failure("boom")
                    else:
                        breaker.snapshot()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        snapshot = breaker.snapshot()
        assert snapshot["state"] in {"open", "closed"}
        assert snapshot["consecutive_failures"] <= 5
