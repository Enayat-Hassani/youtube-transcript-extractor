from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ytx_api.main as main_module
from ytx_api.main import create_app
from ytx_core import (
    AllBackendsFailedError,
    AttemptRecord,
    InvalidInputError,
    LanguageOption,
    NoTranscriptFoundError,
    PlaylistNotSupportedError,
    Segment,
    SourceInfo,
    SourceKind,
    TranscriptDocument,
    TranscriptsDisabledError,
    VideoUnavailableError,
    YtxError,
)


def make_doc() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="dQw4w9WgXcQ",
        language="en",
        language_label="English",
        is_generated=False,
        duration_sec=12.5,
        segments=[
            Segment(start=0.0, end=6.0, text="hello world"),
            Segment(start=6.0, end=12.5, text="second line"),
        ],
        source=SourceInfo(kind=SourceKind.MANUAL_CAPTIONS, backend="unit-test"),
    )


class FakeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.video_calls: list[str] = []
        self.error: Exception | None = None

    def get(
        self,
        url_or_id: str,
        *,
        languages: list[str] | None = None,
        refresh: bool = False,
        translate_to: str | None = None,
    ) -> TranscriptDocument:
        self.calls.append(
            {
                "url_or_id": url_or_id,
                "languages": languages,
                "refresh": refresh,
                "translate_to": translate_to,
            }
        )
        if self.error is not None:
            raise self.error
        return make_doc()

    def list_languages(self, video_id: str) -> list[LanguageOption]:
        self.video_calls.append(video_id)
        return [
            LanguageOption(
                language_code="en", language_label="English", kind=SourceKind.MANUAL_CAPTIONS
            ),
            LanguageOption(
                language_code="de",
                language_label="Deutsch",
                kind=SourceKind.AUTO_CAPTIONS,
                is_translatable=True,
            ),
        ]

    def health(self) -> dict[str, object]:
        return {"backends": [{"backend": "fake", "state": "closed"}]}


CANNED_JOB = {
    "job_id": "job-1",
    "status": "done",
    "options": {"urls": ["https://example.com/a"], "languages": None, "refresh": False},
    "results": [
        {
            "url": "https://example.com/a",
            "ok": True,
            "video_id": "abc123",
            "language": "en",
            "segments": 2,
        }
    ],
    "created_at": 1000.0,
    "started_at": 1000.5,
    "finished_at": 1001.0,
    "error": None,
}

CANNED_JOB_LIST = [
    {
        "job_id": f"job-{index}",
        "status": "done",
        "options": {},
        "created_at": float(index),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    for index in range(1, 6)
]


class FakeRunner:
    def __init__(self) -> None:
        self.submit_calls: list[dict[str, object]] = []
        self.listed_limits: list[int] = []
        self.shutdown_calls: list[bool] = []
        self._counter = 0

    def submit(
        self,
        urls: list[str],
        *,
        languages: list[str] | None = None,
        refresh: bool = False,
    ) -> str:
        self._counter += 1
        self.submit_calls.append({"urls": list(urls), "languages": languages, "refresh": refresh})
        return f"job-{self._counter}"

    def get_job(self, job_id: str) -> dict[str, object] | None:
        if job_id == CANNED_JOB["job_id"]:
            return CANNED_JOB
        return None

    def list_jobs(self, limit: int = 20) -> list[dict[str, object]]:
        self.listed_limits.append(limit)
        return CANNED_JOB_LIST[:limit]

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


class RecordingJobRunner:
    instances: list[RecordingJobRunner] = []

    def __init__(self, service_factory: object, db_path: object, *, max_workers: int = 4) -> None:
        self.service_factory = service_factory
        self.db_path = db_path
        self.max_workers = max_workers
        self.shutdown_calls: list[bool] = []
        RecordingJobRunner.instances.append(self)

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def client(service: FakeService, runner: FakeRunner) -> TestClient:
    return TestClient(create_app(service=service, runner=runner))


def test_json_happy_path(client: TestClient, service: FakeService) -> None:
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    expected = make_doc().model_dump(mode="json") | {"fetched_at": body["fetched_at"]}
    assert body == expected
    assert service.calls[0]["url_or_id"] == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("srt", "application/x-subrip"),
        ("vtt", "text/vtt"),
        ("txt", "text/plain"),
        ("md", "text/plain"),
    ],
)
def test_non_json_formats_content_type(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fmt: str,
    media_type: str,
) -> None:
    seen: dict[str, object] = {}

    def fake_format(doc: TranscriptDocument, out_fmt: str) -> str:
        seen["fmt"] = out_fmt
        return f"{out_fmt} payload"

    monkeypatch.setattr(main_module, "format_transcript", fake_format)
    response = client.get(f"/api/v1/transcripts/dQw4w9WgXcQ?format={fmt}")
    assert response.status_code == 200
    assert response.headers["content-type"] == f"{media_type}; charset=utf-8"
    assert response.text == f"{fmt} payload"
    assert seen["fmt"] == fmt


def test_bad_format_rejected(client: TestClient) -> None:
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ?format=xml")
    assert response.status_code == 422


def test_download_sets_content_disposition(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "format_transcript", lambda doc, fmt: f"{fmt} payload")
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ?format=srt&download=true")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="dQw4w9WgXcQ.srt"'


def test_no_content_disposition_by_default(client: TestClient) -> None:
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert "content-disposition" not in response.headers


def test_comma_separated_languages_reach_service(client: TestClient, service: FakeService) -> None:
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ?languages=en,de")
    assert response.status_code == 200
    assert service.calls[0]["languages"] == ["en", "de"]


def test_missing_languages_pass_none(client: TestClient, service: FakeService) -> None:
    client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert service.calls[0]["languages"] is None


def test_refresh_passthrough_default_false(client: TestClient, service: FakeService) -> None:
    client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert service.calls[0]["refresh"] is False


def test_refresh_passthrough_true(client: TestClient, service: FakeService) -> None:
    client.get("/api/v1/transcripts/dQw4w9WgXcQ?refresh=true")
    assert service.calls[0]["refresh"] is True


ERROR_CASES: list[tuple[Exception, int]] = [
    (InvalidInputError("could not parse"), 400),
    (PlaylistNotSupportedError("playlists unsupported"), 400),
    (VideoUnavailableError("video gone"), 404),
    (TranscriptsDisabledError("captions off"), 404),
    (NoTranscriptFoundError("dQw4w9WgXcQ", ["en"]), 404),
    (YtxError("generic failure"), 400),
]


@pytest.mark.parametrize(("exc", "expected_status"), ERROR_CASES)
def test_error_mapping(
    client: TestClient,
    service: FakeService,
    exc: Exception,
    expected_status: int,
) -> None:
    service.error = exc
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert response.status_code == expected_status
    body = response.json()
    assert body["error"]["type"] == type(exc).__name__
    assert body["error"]["message"] == str(exc)


def test_all_backends_failed_maps_to_502_with_attempts(
    client: TestClient,
    service: FakeService,
) -> None:
    service.error = AllBackendsFailedError(
        attempts=[
            AttemptRecord(backend="alpha", message="timeout", retryable=True),
            AttemptRecord(backend="beta", message="blocked", retryable=False),
        ]
    )
    response = client.get("/api/v1/transcripts/dQw4w9WgXcQ")
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "AllBackendsFailedError"
    assert body["error"]["attempts"] == [
        {"backend": "alpha", "message": "timeout", "retryable": True},
        {"backend": "beta", "message": "blocked", "retryable": False},
    ]


def test_video_languages_endpoint(client: TestClient, service: FakeService) -> None:
    service.calls.clear()
    service.video_calls.clear()
    response = client.get("/api/v1/videos/dQw4w9WgXcQ")
    assert response.status_code == 200
    expected = {
        "video_id": "dQw4w9WgXcQ",
        "languages": [option.model_dump(mode="json") for option in service.list_languages("x")],
    }
    assert response.json() == expected
    assert service.video_calls[0] == "dQw4w9WgXcQ"


def test_health_shape(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]
    assert body["backends"] == [{"backend": "fake", "state": "closed"}]


def test_submit_batch_happy_path(client: TestClient, runner: FakeRunner) -> None:
    response = client.post(
        "/api/v1/transcripts",
        json={
            "urls": ["https://example.com/a", "https://example.com/b"],
            "languages": ["en", "de"],
            "refresh": True,
        },
    )
    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "count": 2}
    assert runner.submit_calls == [
        {
            "urls": ["https://example.com/a", "https://example.com/b"],
            "languages": ["en", "de"],
            "refresh": True,
        }
    ]


def test_submit_batch_defaults(client: TestClient, runner: FakeRunner) -> None:
    response = client.post("/api/v1/transcripts", json={"urls": ["https://example.com/a"]})
    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "count": 1}
    assert runner.submit_calls == [
        {"urls": ["https://example.com/a"], "languages": None, "refresh": False}
    ]


def test_submit_batch_rejects_empty_urls(client: TestClient) -> None:
    response = client.post("/api/v1/transcripts", json={"urls": []})
    assert response.status_code == 422


def test_submit_batch_rejects_over_500_urls(client: TestClient) -> None:
    response = client.post("/api/v1/transcripts", json={"urls": ["https://x"] * 501})
    assert response.status_code == 422


def test_get_job_found(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/job-1")
    assert response.status_code == 200
    assert response.json() == CANNED_JOB


def test_get_job_unknown_returns_404_shape(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/missing")
    assert response.status_code == 404
    assert response.json() == {"error": {"type": "NotFound", "message": "job not found"}}


def test_list_jobs_wraps_in_jobs_key_with_default_limit(
    client: TestClient, runner: FakeRunner
) -> None:
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()["jobs"]
    assert [job["job_id"] for job in jobs] == [f"job-{index}" for index in range(1, 6)]
    assert runner.listed_limits == [20]


def test_list_jobs_honors_limit(client: TestClient, runner: FakeRunner) -> None:
    response = client.get("/api/v1/jobs?limit=2")
    assert response.status_code == 200
    jobs = response.json()["jobs"]
    assert [job["job_id"] for job in jobs] == ["job-1", "job-2"]
    assert runner.listed_limits == [2]


def test_list_jobs_rejects_limit_over_100(client: TestClient) -> None:
    response = client.get("/api/v1/jobs?limit=101")
    assert response.status_code == 422


def test_injected_runner_shutdown_on_lifespan_exit(service: FakeService) -> None:
    fake_runner = FakeRunner()
    with TestClient(create_app(service=service, runner=fake_runner)) as http:
        assert http.get("/health").status_code == 200
    assert fake_runner.shutdown_calls == [False]


@pytest.fixture
def recording_runner_ctor(monkeypatch: pytest.MonkeyPatch) -> type[RecordingJobRunner]:
    RecordingJobRunner.instances.clear()
    monkeypatch.setattr(main_module, "JobRunner", RecordingJobRunner)
    return RecordingJobRunner


def test_default_runner_built_from_env(
    recording_ctor: type[RecordingTranscriptService],
    recording_runner_ctor: type[RecordingJobRunner],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_db = tmp_path / "jobs.sqlite3"
    monkeypatch.setenv("YTX_JOBS_DB_PATH", str(jobs_db))
    monkeypatch.setenv("YTX_MAX_JOB_WORKERS", "7")
    with TestClient(create_app()) as http:
        assert http.get("/health").status_code == 200
    instance = recording_runner_ctor.instances[0]
    assert instance.db_path == str(jobs_db)
    assert instance.max_workers == 7
    service_factory_result = instance.service_factory()
    assert service_factory_result is recording_ctor.instances[0]
    assert instance.shutdown_calls == [False]


def test_default_runner_env_defaults(
    recording_ctor: type[RecordingTranscriptService],
    recording_runner_ctor: type[RecordingJobRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YTX_JOBS_DB_PATH", raising=False)
    monkeypatch.delenv("YTX_MAX_JOB_WORKERS", raising=False)
    with TestClient(create_app()) as http:
        assert http.get("/health").status_code == 200
    instance = recording_runner_ctor.instances[0]
    assert instance.db_path == "./ytx_jobs.sqlite3"
    assert instance.max_workers == 4


class RecordingTranscriptService:
    instances: list[RecordingTranscriptService] = []

    def __init__(self, db_path: str | None = None, *, enable_cache: bool = True) -> None:
        self.db_path = db_path
        self.enable_cache = enable_cache
        self.closed = False
        RecordingTranscriptService.instances.append(self)

    def health(self) -> dict[str, object]:
        return {"backends": []}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recording_ctor(monkeypatch: pytest.MonkeyPatch) -> type[RecordingTranscriptService]:
    RecordingTranscriptService.instances.clear()
    monkeypatch.setattr(main_module, "TranscriptService", RecordingTranscriptService)
    return RecordingTranscriptService


def test_default_service_built_from_env(
    recording_ctor: type[RecordingTranscriptService],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    db_file = tmp_path / "db.sqlite"  # type: ignore[operator]
    monkeypatch.setenv("YTX_DB_PATH", str(db_file))
    monkeypatch.setenv("YTX_DISABLE_CACHE", "TRUE")
    with TestClient(create_app(runner=FakeRunner())) as http:
        assert http.get("/health").status_code == 200
    instance = recording_ctor.instances[0]
    assert instance.db_path == str(db_file)
    assert instance.enable_cache is False
    assert instance.closed is True


def test_default_service_cache_enabled_when_env_unset(
    recording_ctor: type[RecordingTranscriptService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YTX_DB_PATH", raising=False)
    monkeypatch.delenv("YTX_DISABLE_CACHE", raising=False)
    with TestClient(create_app(runner=FakeRunner())) as http:
        assert http.get("/health").status_code == 200
    instance = recording_ctor.instances[0]
    assert instance.db_path is None
    assert instance.enable_cache is True


def test_injected_service_not_rebuilt_by_lifespan(service: FakeService) -> None:
    with TestClient(create_app(service=service, runner=FakeRunner())) as http:
        assert http.get("/health").status_code == 200
