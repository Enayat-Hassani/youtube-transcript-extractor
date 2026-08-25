from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ytx_core.models import TranscriptDocument

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    document_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    PRIMARY KEY (video_id, language)
)
"""


class TranscriptCache:
    """SQLite-backed cache of transcript documents keyed by (video_id, language)."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(
        self,
        video_id: str,
        language: str = "",
        now: float | None = None,
    ) -> TranscriptDocument | None:
        current = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT document_json, expires_at FROM transcripts "
                "WHERE video_id = ? AND language = ?",
                (video_id, language),
            ).fetchone()
            if row is None:
                return None
            document_json, expires_at = row
            if expires_at is not None and expires_at <= current:
                self._conn.execute(
                    "DELETE FROM transcripts WHERE video_id = ? AND language = ?",
                    (video_id, language),
                )
                self._conn.commit()
                return None
            return TranscriptDocument.model_validate_json(document_json)

    def put(
        self,
        doc: TranscriptDocument,
        ttl_seconds: float | None = None,
        now: float | None = None,
        language: str = "",
    ) -> None:
        current = time.time() if now is None else now
        expires_at = None if ttl_seconds is None else current + ttl_seconds
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO transcripts "
                "(video_id, language, kind, document_json, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc.video_id,
                    language,
                    doc.source.kind.value,
                    doc.model_dump_json(),
                    current,
                    expires_at,
                ),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM transcripts")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
