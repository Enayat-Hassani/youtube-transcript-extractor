from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from yt_dlp.utils import DownloadError

import ytx_core.expansion as expansion_module
import ytx_core.service as service_module
from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.errors import InvalidInputError
from ytx_core.expansion import expand_to_video_ids, is_expandable_input
from ytx_core.service import TranscriptService

VALID_IDS = [
    "aaaaaaaaaaa",
    "bbbbbbbbbbb",
    "ccccccccccc",
    "ddddddddddd",
    "eeeeeeeeeee",
]


class FakeBackend(TranscriptBackend):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, request: FetchRequest):
        self.calls += 1
        raise AssertionError("fetch must not be called during expand")


def install_fake_ydl(monkeypatch: pytest.MonkeyPatch, *, info=None, exc=None) -> list[dict]:
    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, opts: dict) -> None:
            self.opts = opts
            record: dict = {"opts": opts}
            calls.append(record)

        def extract_info(self, url: str, download: bool = False) -> dict:
            record = calls[-1]
            record["url"] = url
            record["download"] = download
            if exc is not None:
                raise exc
            assert info is not None
            return info

    monkeypatch.setattr(expansion_module, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    return calls


class TestExpandToVideoIds:
    def test_playlist_entries_filtered_ordered_none_safe(self, monkeypatch):
        entries = [{"id": vid} for vid in VALID_IDS] + [{"id": "short"}, None]
        calls = install_fake_ydl(monkeypatch, info={"id": "PLx", "entries": entries})
        assert expand_to_video_ids("https://youtube.com/playlist?list=PLx") == VALID_IDS
        assert calls[0]["opts"]["extract_flat"] == "in_playlist"
        assert calls[0]["download"] is False

    def test_limit_caps_output(self, monkeypatch):
        entries = [{"id": vid} for vid in VALID_IDS]
        install_fake_ydl(monkeypatch, info={"entries": entries})
        result = expand_to_video_ids("https://youtube.com/playlist?list=PLx", limit=3)
        assert result == VALID_IDS[:3]

    def test_dedupe_keeps_first_occurrence(self, monkeypatch):
        entries = [
            {"id": VALID_IDS[0]},
            {"id": VALID_IDS[1]},
            {"id": VALID_IDS[0]},
            {"id": VALID_IDS[2]},
            {"id": VALID_IDS[1]},
        ]
        install_fake_ydl(monkeypatch, info={"entries": entries})
        result = expand_to_video_ids("https://youtube.com/playlist?list=PLx")
        assert result == [VALID_IDS[0], VALID_IDS[1], VALID_IDS[2]]

    def test_single_video_info_returns_one_id(self, monkeypatch):
        calls = install_fake_ydl(monkeypatch, info={"id": "dQw4w9WgXcQ"})
        assert expand_to_video_ids("dQw4w9WgXcQ") == ["dQw4w9WgXcQ"]
        assert calls[0]["url"] == "dQw4w9WgXcQ"

    def test_empty_entries_raises_no_videos_found(self, monkeypatch):
        install_fake_ydl(monkeypatch, info={"id": "PLx", "entries": []})
        with pytest.raises(InvalidInputError, match="no videos found"):
            expand_to_video_ids("https://youtube.com/playlist?list=PLx")

    def test_download_error_chained_into_invalid_input(self, monkeypatch):
        error = DownloadError("boom")
        install_fake_ydl(monkeypatch, exc=error)
        with pytest.raises(InvalidInputError, match="DownloadError") as excinfo:
            expand_to_video_ids("https://youtube.com/@nope")
        assert excinfo.value.__cause__ is error

    def test_handle_url_passed_through_unchanged(self, monkeypatch):
        entries = [{"id": VALID_IDS[0]}]
        calls = install_fake_ydl(monkeypatch, info={"id": "PLx", "entries": entries})
        expand_to_video_ids("@somechannel/videos")
        assert calls[0]["url"] == "@somechannel/videos"


class TestIsExpandableInput:
    @pytest.mark.parametrize(
        "text",
        [
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/watch?list=xyz",
            "@somechannel/videos",
            "https://www.youtube.com/@somechannel",
            "https://www.youtube.com/channel/UCxyz456abc",
            "https://www.youtube.com/c/somename/videos",
            "https://www.youtube.com/user/someuser",
        ],
    )
    def test_expandable_inputs(self, text: str):
        assert is_expandable_input(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        ],
    )
    def test_non_expandable_inputs(self, text: str):
        assert is_expandable_input(text) is False


class TestServiceExpand:
    def test_direct_id_resolves_without_ytdlp(self, tmp_path: Path, monkeypatch):
        class ExplodingYDL:
            def __init__(self, opts: dict) -> None:
                raise AssertionError("yt-dlp must not be used for direct video ids")

        monkeypatch.setattr(expansion_module, "yt_dlp", SimpleNamespace(YoutubeDL=ExplodingYDL))
        backend = FakeBackend()
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[backend])
        assert service.expand("dQw4w9WgXcQ") == ["dQw4w9WgXcQ"]
        assert backend.calls == 0
        service.close()

    def test_playlist_url_delegates_to_expansion(self, tmp_path: Path, monkeypatch):
        sentinel = ["sentinel-id-1"]
        seen_kwargs: dict = {}

        def fake_expand(url_or_id: str, *, limit: int = 500) -> list[str]:
            seen_kwargs.update(url=url_or_id, limit=limit)
            return sentinel

        monkeypatch.setattr(service_module, "expand_to_video_ids", fake_expand)
        backend = FakeBackend()
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[backend])
        result = service.expand("https://www.youtube.com/playlist?list=PLabc", limit=7)
        assert result is sentinel
        assert seen_kwargs == {
            "url": "https://www.youtube.com/playlist?list=PLabc",
            "limit": 7,
        }
        service.close()

    def test_both_paths_failing_raises_invalid_input(self, tmp_path: Path, monkeypatch):
        def failing_expand(url_or_id: str, *, limit: int = 500) -> list[str]:
            raise InvalidInputError(f"could not expand {url_or_id!r}: DownloadError")

        monkeypatch.setattr(service_module, "expand_to_video_ids", failing_expand)
        service = TranscriptService(tmp_path / "cache.sqlite3", backends=[FakeBackend()])
        with pytest.raises(InvalidInputError):
            service.expand("definitely-not-a-video-ref")
        service.close()
