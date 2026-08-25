"""Offline CLI tests backed by a fake TranscriptService."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import ytx_core.cleanup as cleanup_pkg
from ytx_cli import main as cli_main
from ytx_core.doc import VideoMetadata
from ytx_core.errors import TranscriptsDisabledError
from ytx_core.models import (
    LanguageOption,
    Segment,
    SourceInfo,
    SourceKind,
    TranscriptDocument,
)


def _document(video_id: str = "dQw4w9WgXcQ") -> TranscriptDocument:
    return TranscriptDocument(
        video_id=video_id,
        language="en",
        language_label="English",
        is_generated=False,
        duration_sec=4.0,
        segments=[
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.0, end=4.0, text="world"),
        ],
        source=SourceInfo(kind=SourceKind.MANUAL_CAPTIONS, backend="youtube-transcript-api"),
    )


def _video_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("v=")[-1]


class FakeService:
    def __init__(self) -> None:
        self.init_kwargs_list: list[dict] = []
        self.get_calls: list[dict] = []
        self.lang_calls: list[str] = []
        self.fail_urls: set[str] = set()

    @property
    def init_kwargs(self) -> dict:
        return self.init_kwargs_list[-1]

    def factory(self, **kwargs):
        self.init_kwargs_list.append(kwargs)
        return self

    def get(self, url, languages=None, refresh=False, translate_to=None):
        self.get_calls.append(
            {
                "url": url,
                "languages": languages,
                "refresh": refresh,
                "translate_to": translate_to,
            }
        )
        if url in self.fail_urls:
            raise TranscriptsDisabledError(f"transcripts disabled for {url}")
        return _document(video_id=_video_id(url))

    def list_languages(self, url):
        self.lang_calls.append(url)
        return [
            LanguageOption(
                language_code="en",
                language_label="English",
                kind=SourceKind.MANUAL_CAPTIONS,
                is_translatable=False,
            ),
            LanguageOption(
                language_code="de",
                language_label="German",
                kind=SourceKind.AUTO_CAPTIONS,
                is_translatable=True,
            ),
        ]

    def health(self):
        return {
            "backends": [
                {"backend": "youtube-transcript-api", "state": "closed", "consecutive_failures": 0},
                {"backend": "timedtext", "state": "open", "consecutive_failures": 3},
            ]
        }


def _fake_format(document, fmt):
    if fmt == "json":
        return document.model_dump_json()
    return f"FAKE-{fmt}"


@pytest.fixture
def fake_service(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(cli_main, "TranscriptService", service.factory)
    monkeypatch.setattr(cli_main, "format_transcript", _fake_format)
    return service


@pytest.fixture
def runner():
    return CliRunner()


def test_get_prints_json_to_stdout_and_status_to_stderr(fake_service, runner, tmp_path):
    db = tmp_path / "cache.sqlite3"
    result = runner.invoke(cli_main.app, ["get", "https://youtu.be/dQw4w9WgXcQ", "--db", str(db)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["video_id"] == "dQw4w9WgXcQ"
    assert payload["language"] == "en"
    assert [segment["text"] for segment in payload["segments"]] == ["Hello", "world"]
    assert payload["source"]["backend"] == "youtube-transcript-api"
    assert "dQw4w9WgXcQ" in result.stderr
    assert "[ok]" in result.stderr


def test_get_srt_output_creates_parent_dirs(fake_service, runner, tmp_path):
    target = tmp_path / "nested" / "dir" / "out.srt"
    result = runner.invoke(
        cli_main.app, ["get", "vid123", "--format", "srt", "--output", str(target)]
    )

    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "FAKE-srt"
    assert result.stdout == ""


def test_get_error_path_exits_1_with_err_marker(fake_service, runner, tmp_path):
    fake_service.fail_urls.add("badvid")
    result = runner.invoke(cli_main.app, ["get", "badvid", "--db", str(tmp_path / "c.sqlite3")])

    assert result.exit_code == 1
    assert "[err]" in result.stderr
    assert "transcripts disabled" in result.stderr


def test_get_forwards_parsed_languages_and_refresh(fake_service, runner, tmp_path):
    db = tmp_path / "cache.sqlite3"
    result = runner.invoke(
        cli_main.app,
        ["get", "abc123", "--lang", "en,de", "--refresh", "--db", str(db)],
    )

    assert result.exit_code == 0
    assert fake_service.get_calls == [
        {"url": "abc123", "languages": ["en", "de"], "refresh": True, "translate_to": None}
    ]
    assert fake_service.init_kwargs == {"db_path": db, "enable_cache": True}


def test_batch_mixed_results_write_files_and_exit_1(fake_service, runner, tmp_path):
    input_file = tmp_path / "urls.txt"
    input_file.write_text(
        "# comment line\nhttps://youtu.be/vidA\n\n   \nhttps://youtu.be/vidB\nvidC\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "out"
    fake_service.fail_urls.add("https://youtu.be/vidB")

    result = runner.invoke(
        cli_main.app,
        ["batch", str(input_file), "--outdir", str(outdir), "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 1
    payload_a = json.loads((outdir / "vidA.json").read_text(encoding="utf-8"))
    assert payload_a["video_id"] == "vidA"
    assert (outdir / "vidC.json").exists()
    assert not (outdir / "vidB.json").exists()
    assert "Done: 2 ok, 1 failed" in result.stdout
    assert [call["url"] for call in fake_service.get_calls] == [
        "https://youtu.be/vidA",
        "https://youtu.be/vidB",
        "vidC",
    ]


def test_batch_all_success_exits_0(fake_service, runner, tmp_path):
    input_file = tmp_path / "urls.txt"
    input_file.write_text("vidA\nvidC\n", encoding="utf-8")
    outdir = tmp_path / "out"

    result = runner.invoke(
        cli_main.app,
        ["batch", str(input_file), "--outdir", str(outdir), "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 0
    assert sorted(path.name for path in outdir.iterdir()) == ["vidA.json", "vidC.json"]
    assert "Done: 2 ok, 0 failed" in result.stdout


def test_langs_renders_table(fake_service, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        ["langs", "https://youtu.be/langvid", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 0
    assert "Available transcripts" in result.stdout
    assert "en" in result.stdout
    assert "de" in result.stdout
    assert fake_service.lang_calls == ["https://youtu.be/langvid"]


def test_health_renders_backend_rows(fake_service, runner, tmp_path):
    result = runner.invoke(cli_main.app, ["health", "--db", str(tmp_path / "c.sqlite3")])

    assert result.exit_code == 0
    assert "backend" in result.stdout
    assert "youtube-transcript-api" in result.stdout
    assert "timedtext" in result.stdout
    assert "open" in result.stdout


def test_version_flag(runner):
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ytx 0.1.0"


FAKE_DOC_MD = "# Fake Doc\n\nContent.\n"


@pytest.fixture(autouse=True)
def no_sponsorblock_network(monkeypatch):
    """The CLI suite must stay offline."""
    monkeypatch.setattr(cleanup_pkg, "fetch_ranges", lambda *a, **k: [])


@pytest.fixture
def fake_doc_pipeline(monkeypatch):
    captured: dict = {}
    sentinel_metadata = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Fake",
        description="Sponsor: https://example.com/aff/go\n\nReal prose about back testing.",
    )

    def fake_fetch(video_id: str):
        captured["video_id"] = video_id
        return sentinel_metadata

    def fake_compose(metadata, document, *, notes=None, screen=()):
        captured["composed_for"] = document.video_id
        captured["description"] = metadata.description
        captured["notes"] = notes
        captured["screen"] = list(screen)
        return FAKE_DOC_MD

    monkeypatch.setattr(cli_main, "fetch_video_metadata", fake_fetch)
    monkeypatch.setattr(cli_main, "compose_markdown_doc", fake_compose)
    return captured


def test_doc_prints_markdown_to_stdout_and_status_to_stderr(
    fake_service, fake_doc_pipeline, runner, tmp_path
):
    db = tmp_path / "cache.sqlite3"
    result = runner.invoke(
        cli_main.app,
        ["doc", "https://youtu.be/dQw4w9WgXcQ", "--db", str(db)],
    )

    assert result.exit_code == 0
    assert FAKE_DOC_MD in result.stdout
    assert fake_doc_pipeline["video_id"] == "dQw4w9WgXcQ"
    assert fake_doc_pipeline["composed_for"] == "dQw4w9WgXcQ"
    assert "[ok] doc dQw4w9WgXcQ" in result.stderr


def test_doc_output_file_creates_parent_dirs(fake_service, fake_doc_pipeline, runner, tmp_path):
    target = tmp_path / "nested" / "dir" / "out.md"
    result = runner.invoke(cli_main.app, ["doc", "dQw4w9WgXcQ", "--output", str(target)])

    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == FAKE_DOC_MD
    assert result.stdout == ""


def test_doc_error_path_exits_1_with_err_marker(fake_service, fake_doc_pipeline, runner, tmp_path):
    fake_service.fail_urls.add("badvideo001")
    result = runner.invoke(
        cli_main.app,
        ["doc", "badvideo001", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 1
    assert "[err]" in result.stderr
    assert "transcripts disabled" in result.stderr


def test_doc_forwards_parsed_languages_and_refresh(
    fake_service, fake_doc_pipeline, runner, tmp_path
):
    db = tmp_path / "cache.sqlite3"
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--lang", "en,de", "--refresh", "--db", str(db)],
    )

    assert result.exit_code == 0
    assert fake_service.get_calls == [
        {
            "url": "dQw4w9WgXcQ",
            "languages": ["en", "de"],
            "refresh": True,
            "translate_to": None,
        }
    ]
    assert fake_service.init_kwargs == {"db_path": db, "enable_cache": True}


def test_doc_applies_cleanup_by_default(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app, ["doc", "dQw4w9WgXcQ", "--db", str(tmp_path / "c.sqlite3")]
    )

    assert result.exit_code == 0
    assert "example.com" not in fake_doc_pipeline["description"]
    assert fake_doc_pipeline["notes"]


def test_doc_domain_flag_selects_the_pack(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--domain", "trading", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 0
    assert "backtesting" in fake_doc_pipeline["description"]


def test_doc_domain_none_skips_domain_packs(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--domain", "none", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 0
    assert "back testing" in fake_doc_pipeline["description"]


def test_doc_unknown_domain_exits_1(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--domain", "nonsense", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 1
    assert "[err]" in result.stderr


def test_doc_no_clean_leaves_everything_raw(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--no-clean", "--db", str(tmp_path / "c.sqlite3")],
    )

    assert result.exit_code == 0
    assert "example.com" in fake_doc_pipeline["description"]
    assert "back testing" in fake_doc_pipeline["description"]
    assert fake_doc_pipeline["notes"] == []


def test_doc_no_fix_terms_keeps_vocabulary(fake_service, fake_doc_pipeline, runner, tmp_path):
    result = runner.invoke(
        cli_main.app,
        [
            "doc", "dQw4w9WgXcQ", "--no-fix-terms", "--domain", "trading",
            "--db", str(tmp_path / "c.sqlite3"),
        ],
    )

    assert result.exit_code == 0
    # description links still stripped, but vocabulary untouched
    assert "example.com" not in fake_doc_pipeline["description"]
    assert "back testing" in fake_doc_pipeline["description"]


def test_packs_command_lists_bundled_packs(runner):
    result = runner.invoke(cli_main.app, ["packs"])

    assert result.exit_code == 0
    assert "trading" in result.stdout
    assert "general" in result.stdout


def test_doc_index_writes_file_and_prints_map(fake_service, fake_doc_pipeline, runner, tmp_path):
    target = tmp_path / "out.md"
    result = runner.invoke(
        cli_main.app,
        ["doc", "dQw4w9WgXcQ", "--index", "--output", str(target), "--db", str(tmp_path / "c.db")],
    )

    assert result.exit_code == 0
    # The document went to disk, not to stdout.
    assert target.read_text(encoding="utf-8") == FAKE_DOC_MD
    assert FAKE_DOC_MD not in result.stdout
    assert f"file:     {target}" in result.stdout
    assert "tokens if read in full" in result.stdout


def test_doc_index_defaults_to_video_id_file(fake_service, fake_doc_pipeline, runner, tmp_path,
                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli_main.app, ["doc", "dQw4w9WgXcQ", "--index", "--no-cache"])

    assert result.exit_code == 0
    assert (tmp_path / "dQw4w9WgXcQ.md").is_file()
