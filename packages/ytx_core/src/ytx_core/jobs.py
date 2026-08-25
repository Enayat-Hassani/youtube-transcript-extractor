from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from ytx_core.service import TranscriptService

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    options_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    results_json TEXT NOT NULL DEFAULT '[]',
    error TEXT
)
"""

_JOB_COLUMNS = (
    "job_id",
    "status",
    "options_json",
    "created_at",
    "started_at",
    "finished_at",
    "results_json",
    "error",
)

_ACTIVE_STATUSES = frozenset({"pending", "running"})


def _row_to_dict(row: tuple, *, with_results: bool) -> dict:
    job = dict(zip(_JOB_COLUMNS, row, strict=True))
    job["options"] = json.loads(job.pop("options_json"))
    if with_results:
        job["results"] = json.loads(job.pop("results_json"))
    else:
        del job["results_json"]
    return job


class JobStore:
    """SQLite-backed store of extraction job status rows."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def create_job(self, job_id: str, options: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, status, options_json, created_at, results_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, "pending", json.dumps(options), time.time(), "[]"),
            )
            self._conn.commit()

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (time.time(), job_id),
            )
            self._conn.commit()

    def finish(
        self, job_id: str, status: str, results: list[dict], error: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, results_json = ?, error = ? "
                "WHERE job_id = ?",
                (status, time.time(), json.dumps(results), error, job_id),
            )
            self._conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_JOB_COLUMNS)} FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row, with_results=True)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_JOB_COLUMNS)} FROM jobs "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(row, with_results=False) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JobRunner:
    """Runs extraction jobs on a bounded thread pool; one worker per job."""

    def __init__(
        self,
        service_factory: Callable[[], TranscriptService],
        db_path: str | Path,
        *,
        max_workers: int = 4,
    ) -> None:
        self._service_factory = service_factory
        self._store = JobStore(db_path)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        urls: Sequence[str],
        *,
        languages: Sequence[str] | None = None,
        refresh: bool = False,
    ) -> str:
        job_id = uuid4().hex
        options = {
            "urls": list(urls),
            "languages": list(languages) if languages else None,
            "refresh": refresh,
        }
        self._store.create_job(job_id, options)
        self._executor.submit(self._run_job, job_id, options["urls"], options["languages"], refresh)
        return job_id

    def _run_job(
        self,
        job_id: str,
        urls: list[str],
        languages: list[str] | None,
        refresh: bool,
    ) -> None:
        self._store.mark_running(job_id)
        results: list[dict] = []
        try:
            service = self._service_factory()
            for url in urls:
                try:
                    doc = service.get(
                        url,
                        languages=list(languages) if languages else None,
                        refresh=refresh,
                    )
                    results.append(
                        {
                            "url": url,
                            "ok": True,
                            "video_id": doc.video_id,
                            "language": doc.language,
                            "segments": len(doc.segments),
                        }
                    )
                except Exception as exc:
                    results.append({"url": url, "ok": False, "error": str(exc)})
        except Exception as exc:
            self._store.finish(job_id, "error", results, error=str(exc))
            return
        if all(result["ok"] for result in results):
            status = "done"
        elif not any(result["ok"] for result in results):
            status = "error"
        else:
            status = "partial"
        self._store.finish(job_id, status, results)

    def get_job(self, job_id: str) -> dict | None:
        return self._store.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        return self._store.list_jobs(limit)

    def wait_for(self, job_id: str, timeout: float | None = None, poll_s: float = 0.05) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self._store.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] not in _ACTIVE_STATUSES:
                return job
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
            time.sleep(poll_s)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
