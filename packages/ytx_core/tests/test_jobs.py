from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from ytx_core.errors import TranscriptsDisabledError
from ytx_core.jobs import JobRunner, JobStore
from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument


def make_document(video_id: str, segment_count: int = 3) -> TranscriptDocument:
    segments = [
        Segment(start=float(i), end=float(i + 1.0), text=f"line {i}") for i in range(segment_count)
    ]
    return TranscriptDocument(
        video_id=video_id,
        language="en",
        segments=segments,
        source=SourceInfo(kind=SourceKind.AUTO_CAPTIONS, backend="fake"),
    )


class FakeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url_or_id: str,
        *,
        languages: Sequence[str] | None = None,
        refresh: bool = False,
    ) -> TranscriptDocument:
        self.calls.append(url_or_id)
        if "ok" not in url_or_id:
            raise TranscriptsDisabledError("creator disabled captions")
        return make_document(url_or_id)


class RecordingFactory:
    def __init__(self) -> None:
        self.services: list[FakeService] = []

    def __call__(self) -> FakeService:
        service = FakeService()
        self.services.append(service)
        return service


@pytest.fixture
def factory() -> RecordingFactory:
    return RecordingFactory()


@pytest.fixture
def runner(factory: RecordingFactory, tmp_path: Path) -> Iterator[JobRunner]:
    instance = JobRunner(factory, tmp_path / "jobs.sqlite3")
    yield instance
    instance.shutdown(wait=True)


class TestSuccessfulRun:
    def test_done_with_per_url_metadata(self, runner: JobRunner, factory: RecordingFactory):
        urls = ["https://youtu.be/ok-one", "https://youtu.be/ok-two"]
        job_id = runner.submit(urls)
        job = runner.wait_for(job_id, timeout=5.0)
        assert job["status"] == "done"
        assert job["error"] is None
        assert [result["url"] for result in job["results"]] == urls
        assert all(result["ok"] for result in job["results"])
        first = job["results"][0]
        assert first["video_id"] == "https://youtu.be/ok-one"
        assert first["language"] == "en"
        assert first["segments"] == 3
        assert len(factory.services) == 1
        assert factory.services[0].calls == urls


class TestPartialRun:
    def test_mixed_success_failure_is_partial(self, runner: JobRunner):
        job_id = runner.submit(["ok-1", "bad-video", "ok-2"])
        job = runner.wait_for(job_id, timeout=5.0)
        assert job["status"] == "partial"
        failed = job["results"][1]
        assert failed["ok"] is False
        assert "creator disabled captions" in failed["error"]
        assert job["results"][0]["ok"] is True
        assert job["results"][2]["ok"] is True


class TestErrorRun:
    def test_all_failures_is_error(self, runner: JobRunner):
        job_id = runner.submit(["nope-1", "nope-2"])
        job = runner.wait_for(job_id, timeout=5.0)
        assert job["status"] == "error"
        assert all(not result["ok"] for result in job["results"])
        assert job["error"] is None


class TestLookup:
    def test_unknown_job_returns_none(self, runner: JobRunner):
        assert runner.get_job("missing-job-id") is None


class TestListJobs:
    def test_newest_first_with_limit_and_status(self, runner: JobRunner, factory: RecordingFactory):
        ids = [runner.submit([f"https://youtu.be/ok-{i}"]) for i in range(5)]
        for job_id in ids:
            runner.wait_for(job_id, timeout=5.0)
        jobs = runner.list_jobs(limit=3)
        assert [job["job_id"] for job in jobs] == list(reversed(ids))[:3]
        assert all(job["status"] == "done" for job in jobs)
        assert all("results" not in job for job in jobs)
        assert len(factory.services) == 5


class TestConcurrency:
    def test_three_concurrent_jobs_complete(self, runner: JobRunner):
        ids = [
            runner.submit([f"https://youtu.be/ok-a{i}", f"https://youtu.be/ok-b{i}"])
            for i in range(3)
        ]
        jobs = [runner.wait_for(job_id, timeout=10.0) for job_id in ids]
        assert all(job["status"] == "done" for job in jobs)
        assert all(len(job["results"]) == 2 for job in jobs)


class TestWaitForTimeout:
    def test_times_out_while_job_running(self, tmp_path: Path):
        class SlowService(FakeService):
            def get(
                self,
                url_or_id: str,
                *,
                languages: Sequence[str] | None = None,
                refresh: bool = False,
            ) -> TranscriptDocument:
                time.sleep(0.2)
                return super().get(url_or_id, languages=languages, refresh=refresh)

        slow_runner = JobRunner(SlowService, tmp_path / "slow.sqlite3")
        try:
            job_id = slow_runner.submit(["ok-slow"])
            with pytest.raises(TimeoutError):
                slow_runner.wait_for(job_id, timeout=0.01, poll_s=0.01)
        finally:
            slow_runner.shutdown(wait=True)


class TestJobStoreLifecycle:
    def test_create_run_finish_roundtrip(self, tmp_path: Path):
        store = JobStore(tmp_path / "store.sqlite3")
        store.create_job("j1", {"urls": ["u1"], "refresh": False})
        job = store.get("j1")
        assert job is not None
        assert job["status"] == "pending"
        assert job["options"] == {"urls": ["u1"], "refresh": False}
        assert job["results"] == []
        assert job["started_at"] is None
        assert job["finished_at"] is None
        store.mark_running("j1")
        running = store.get("j1")
        assert running is not None
        assert running["status"] == "running"
        assert running["started_at"] is not None
        store.finish("j1", "partial", [{"url": "u1", "ok": False}], error="boom")
        finished = store.get("j1")
        assert finished is not None
        assert finished["status"] == "partial"
        assert finished["results"] == [{"url": "u1", "ok": False}]
        assert finished["error"] == "boom"
        assert finished["finished_at"] is not None
        store.close()
