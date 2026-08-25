from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ytx_core.cache import TranscriptCache
from ytx_core.models import SourceInfo, SourceKind, TranscriptDocument


def make_document(video_id: str = "dQw4w9WgXcQ", kind: SourceKind = SourceKind.AUTO_CAPTIONS):
    return TranscriptDocument(
        video_id=video_id,
        language="en",
        language_label="English",
        is_generated=kind != SourceKind.MANUAL_CAPTIONS,
        duration_sec=3.0,
        segments=[],
        source=SourceInfo(kind=kind, backend="captions_api"),
        fetched_at=datetime.now(UTC),
    )


def row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    finally:
        conn.close()


class TestRoundTrip:
    def test_put_then_get(self, tmp_path: Path):
        cache = TranscriptCache(tmp_path / "cache.sqlite3")
        doc = make_document()
        cache.put(doc)
        assert cache.get(doc.video_id) == doc
        cache.close()

    def test_language_is_part_of_the_key(self, tmp_path: Path):
        cache = TranscriptCache(tmp_path / "cache.sqlite3")
        doc = make_document()
        cache.put(doc, language="en")
        assert cache.get(doc.video_id, "en") == doc
        assert cache.get(doc.video_id, "de") is None
        assert cache.get(doc.video_id) is None
        cache.close()


class TestExpiry:
    def test_ttl_expires_via_injected_now(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        cache = TranscriptCache(db_path)
        doc = make_document()
        cache.put(doc, ttl_seconds=10.0, now=1000.0)
        assert cache.get(doc.video_id, now=1009.999) == doc
        assert cache.get(doc.video_id, now=1011.0) is None
        cache.close()
        assert row_count(db_path) == 0

    def test_forever_ttl_survives_later_reads(self, tmp_path: Path):
        cache = TranscriptCache(tmp_path / "cache.sqlite3")
        doc = make_document()
        cache.put(doc, ttl_seconds=None, now=1000.0)
        assert cache.get(doc.video_id, now=10_000_000.0) == doc
        cache.close()

    def test_expired_rows_are_deleted_on_read(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        cache = TranscriptCache(db_path)
        cache.put(make_document(), ttl_seconds=5.0, now=100.0)
        cache.get("dQw4w9WgXcQ", now=200.0)
        cache.close()
        assert row_count(db_path) == 0


class TestLifecycle:
    def test_clear(self, tmp_path: Path):
        cache = TranscriptCache(tmp_path / "cache.sqlite3")
        cache.put(make_document())
        cache.clear()
        assert cache.get("dQw4w9WgXcQ") is None
        cache.close()

    def test_reopen_sees_persisted_rows(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite3"
        first = TranscriptCache(db_path)
        doc = make_document()
        first.put(doc)
        first.close()
        second = TranscriptCache(db_path)
        assert second.get(doc.video_id) == doc
        second.close()

    def test_creates_parent_directories(self, tmp_path: Path):
        db_path = tmp_path / "nested" / "dir" / "cache.sqlite3"
        cache = TranscriptCache(db_path)
        cache.put(make_document())
        assert db_path.exists()
        cache.close()
