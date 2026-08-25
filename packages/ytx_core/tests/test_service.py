from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from helpers import make_document
from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.errors import AllBackendsFailedError, BackendError
from ytx_core.models import LanguageOption, SourceInfo, SourceKind, TranscriptDocument
from ytx_core.service import AUTO_TTL_SECONDS, TranscriptService


class FakeBackend(TranscriptBackend):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.next_doc = make_document()

    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        self.calls += 1
        return self.next_doc


def expires_at_rows(db_path: Path) -> list[float | None]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [row[0] for row in conn.execute("SELECT expires_at FROM transcripts")]
    finally:
        conn.close()


class TestGet:
    def test_second_get_hits_cache(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        backend = FakeBackend()
        service = TranscriptService(db_path, backends=[backend])
        first = service.get("dQw4w9WgXcQ")
        second = service.get("dQw4w9WgXcQ")
        assert backend.calls == 1
        assert second == first
        service.close()

    def test_refresh_forces_refetch(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        backend = FakeBackend()
        service = TranscriptService(db_path, backends=[backend])
        service.get("dQw4w9WgXcQ")
        service.get("dQw4w9WgXcQ", refresh=True)
        assert backend.calls == 2
        service.close()

    def test_language_scoped_cache_keys(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        backend = FakeBackend()
        service = TranscriptService(db_path, backends=[backend])
        service.get("https://youtu.be/dQw4w9WgXcQ", languages=("en",))
        service.get("dQw4w9WgXcQ", languages=("en",))
        assert backend.calls == 1
        service.get("dQw4w9WgXcQ", languages=("de",))
        assert backend.calls == 2
        service.close()

    def test_auto_captions_ttl_vs_manual_forever(self, tmp_path: Path):
        auto_db = tmp_path / "auto.sqlite3"
        manual_db = tmp_path / "manual.sqlite3"
        auto_backend = FakeBackend()
        service = TranscriptService(auto_db, backends=[auto_backend])
        service.get("dQw4w9WgXcQ")
        rows = expires_at_rows(auto_db)
        assert len(rows) == 1
        assert rows[0] is not None
        service.close()

        manual_backend = FakeBackend()
        manual_backend.next_doc = make_document(kind=SourceKind.MANUAL_CAPTIONS)
        service = TranscriptService(manual_db, backends=[manual_backend])
        service.get("dQw4w9WgXcQ")
        assert expires_at_rows(manual_db) == [None]
        service.close()


class TestCacheDisabled:
    def test_enable_cache_false_skips_cache(self, tmp_path: Path):
        backend = FakeBackend()
        service = TranscriptService(None, enable_cache=False, backends=[backend])
        service.get("dQw4w9WgXcQ")
        service.get("dQw4w9WgXcQ")
        assert backend.calls == 2
        assert service.health()["backends"][0]["backend"] == "fake"
        service.close()


class TestListLanguages:
    def test_delegates_to_listing_backend(self, tmp_path: Path):
        class ListingBackend(FakeBackend):
            supports_language_listing = True

            def list_transcripts(self, video_id: str) -> list[LanguageOption]:
                self.calls += 1
                return [
                    LanguageOption(
                        language_code="en",
                        language_label="English",
                        kind=self.next_doc.source.kind,
                        is_translatable=True,
                    )
                ]

        backend = ListingBackend()
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[backend])
        options = service.list_languages("dQw4w9WgXcQ")
        assert [option.language_code for option in options] == ["en"]
        assert options[0].kind.value == "auto_captions"
        service.close()

    def test_backend_without_listing_support_is_skipped(self, tmp_path: Path):
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[FakeBackend()])
        with pytest.raises(AllBackendsFailedError):
            service.list_languages("dQw4w9WgXcQ")

    def test_transient_failure_collects_attempts(self, tmp_path: Path):
        class FailingListingBackend(FakeBackend):
            supports_language_listing = True

            def list_transcripts(self, video_id: str) -> list[LanguageOption]:
                raise BackendError("fake", "boom")

        service = TranscriptService(
            tmp_path / "cache.sqlite3",
            backends=[FailingListingBackend()],
        )
        with pytest.raises(AllBackendsFailedError) as excinfo:
            service.list_languages("dQw4w9WgXcQ")
        assert [attempt.backend for attempt in excinfo.value.attempts] == ["fake"]


class TestHealth:
    def test_shape(self, tmp_path: Path):
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[FakeBackend()])
        health = service.health()
        assert set(health) == {"backends"}
        assert health["backends"][0]["backend"] == "fake"
        assert health["backends"][0]["state"] == "closed"
        service.close()


def test_auto_ttl_constant():
    assert AUTO_TTL_SECONDS == 30 * 86400


def test_translate_to_uses_separate_cache_bucket(tmp_path, monkeypatch):
    from helpers import make_document

    calls = []

    class CountingBackend(TranscriptBackend):
        name = "counting"

        def fetch(self, request):
            calls.append(request.translate_to)
            doc = make_document(kind=SourceKind.AUTO_CAPTIONS)
            if request.translate_to:
                doc = doc.model_copy(
                    update={
                        "language": request.translate_to,
                        "source": SourceInfo(kind=SourceKind.AUTO_CAPTIONS, backend="captions_api"),
                    }
                )
            return doc

    svc = TranscriptService(db_path=tmp_path / "c.sqlite3", backends=[CountingBackend()])
    first = svc.get("dQw4w9WgXcQ", translate_to="en")
    second = svc.get("dQw4w9WgXcQ", translate_to="en")
    plain = svc.get("dQw4w9WgXcQ")
    assert len(calls) == 2
    assert second.language == "en" and first.language == "en"
    assert plain.language == "en"
