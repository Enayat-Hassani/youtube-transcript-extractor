from __future__ import annotations

from collections.abc import Callable

import pytest

from helpers import make_document
from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.cascade import Cascade
from ytx_core.errors import (
    AllBackendsFailedError,
    BackendError,
    NoTranscriptFoundError,
    TranscriptsDisabledError,
)
from ytx_core.models import TranscriptDocument


class FakeBackend(TranscriptBackend):
    def __init__(
        self,
        name: str,
        behavior: Callable[[FetchRequest], TranscriptDocument | Exception],
    ) -> None:
        self.name = name
        self.behavior = behavior
        self.calls = 0

    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        self.calls += 1
        result = self.behavior(request)
        if isinstance(result, Exception):
            raise result
        return result


def ok_doc(video_id: str = "dQw4w9WgXcQ") -> TranscriptDocument:
    return make_document(video_id=video_id)


def make_request() -> FetchRequest:
    return FetchRequest("dQw4w9WgXcQ")


class TestSuccessPath:
    def test_first_backend_success_records_breaker_success(self):
        first = FakeBackend("first", lambda request: ok_doc())
        second = FakeBackend("second", lambda request: pytest.fail("should not be called"))
        cascade = Cascade([first, second])
        doc = cascade.fetch(make_request())
        assert doc.video_id == "dQw4w9WgXcQ"
        assert first.calls == 1
        health = {entry["backend"]: entry for entry in cascade.health()}
        assert health["first"]["state"] == "closed"
        assert health["first"]["consecutive_failures"] == 0


class TestFallthrough:
    def test_retryable_failure_falls_through_and_counts(self):
        failing = FakeBackend("failing", lambda request: BackendError("failing", "network down"))
        succeeding = FakeBackend("succeeding", lambda request: ok_doc())
        cascade = Cascade([failing, succeeding])
        doc = cascade.fetch(make_request())
        assert doc.video_id == "dQw4w9WgXcQ"
        health = {entry["backend"]: entry for entry in cascade.health()}
        assert health["failing"]["consecutive_failures"] == 1
        assert health["succeeding"]["consecutive_failures"] == 0

    def test_definitive_error_falls_through_without_breaker_effect(self):
        cases = [
            TranscriptsDisabledError(),
            NoTranscriptFoundError("dQw4w9WgXcQ", ["en"]),
        ]
        for definitive in cases:
            failing = FakeBackend("a", lambda request, err=definitive: err)
            succeeding = FakeBackend("b", lambda request: ok_doc())
            cascade = Cascade([failing, succeeding])
            doc = cascade.fetch(make_request())
            assert doc.video_id == "dQw4w9WgXcQ"
            health = {entry["backend"]: entry for entry in cascade.health()}
            assert health["a"]["consecutive_failures"] == 0

    def test_non_retryable_backend_error_still_counts_toward_breaker(self):
        failing = FakeBackend(
            "stubbed",
            lambda request: BackendError("stubbed", "not implemented", retryable=False),
        )
        succeeding = FakeBackend("ok", lambda request: ok_doc())
        cascade = Cascade([failing, succeeding])
        doc = cascade.fetch(make_request())
        assert doc.video_id == "dQw4w9WgXcQ"
        health = {entry["backend"]: entry for entry in cascade.health()}
        assert health["stubbed"]["consecutive_failures"] == 1


class TestOpenBreakerSkips:
    def test_open_breaker_skips_fetch_entirely(self):
        backend = FakeBackend("blocked", lambda request: BackendError("blocked", "boom"))
        cascade = Cascade([backend], failure_threshold=1)
        with pytest.raises(AllBackendsFailedError):
            cascade.fetch(make_request())
        assert backend.calls == 1
        with pytest.raises(AllBackendsFailedError):
            cascade.fetch(make_request())
        assert backend.calls == 1


class TestExhaustion:
    def test_all_fail_raises_with_attempts(self):
        a = FakeBackend("a", lambda request: NoTranscriptFoundError("dQw4w9WgXcQ", ["en"]))
        b = FakeBackend("b", lambda request: BackendError("b", "timeout"))
        cascade = Cascade([a, b])
        with pytest.raises(AllBackendsFailedError) as excinfo:
            cascade.fetch(make_request())
        attempts = excinfo.value.attempts
        assert len(attempts) == 2
        assert [attempt.backend for attempt in attempts] == ["a", "b"]
        assert all(not attempt.ok for attempt in attempts)
        assert all(attempt.message for attempt in attempts)
        assert attempts[0].retryable is False
        assert attempts[1].retryable is True

    def test_no_backends_at_all(self):
        cascade = Cascade([])
        with pytest.raises(AllBackendsFailedError) as excinfo:
            cascade.fetch(make_request())
        assert excinfo.value.attempts == []


class TestHealthShape:
    def test_health_lists_one_snapshot_per_backend(self):
        cascade = Cascade(
            [FakeBackend("one", lambda r: ok_doc()), FakeBackend("two", lambda r: ok_doc())]
        )
        health = cascade.health()
        assert [entry["backend"] for entry in health] == ["one", "two"]
        assert all(entry["state"] == "closed" for entry in health)
