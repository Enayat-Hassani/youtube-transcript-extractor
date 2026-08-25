from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from yt_dlp.utils import DownloadError

import ytx_core.doc as doc_module
from ytx_core.doc import (
    Chapter,
    VideoMetadata,
    compose_index,
    compose_markdown_doc,
    fetch_video_metadata,
)
from ytx_core.errors import BackendError, VideoUnavailableError
from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument

VIDEO_ID = "dQw4w9WgXcQ"

CANNED_INFO = {
    "id": VIDEO_ID,
    "title": "Sample: Talk",
    "uploader": "Some Uploader",
    "channel": "Some Channel",
    "upload_date": "20240131",
    "duration": 123.4,
    "description": "A description.",
    "webpage_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
    "thumbnail": "https://example.com/thumb.jpg",
    "chapters": [
        {"title": "Intro", "start_time": 0},
        {"title": "Main", "start_time": 90.5},
    ],
}


def install_fake_ydl(monkeypatch: pytest.MonkeyPatch, *, info=None, exc=None) -> list[dict]:
    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, opts: dict) -> None:
            self.opts = opts
            record: dict = {"opts": opts}
            calls.append(record)

        def extract_info(self, url: str, download: bool = False) -> dict:
            calls[-1]["url"] = url
            calls[-1]["download"] = download
            if exc is not None:
                raise exc
            assert info is not None
            return info

    monkeypatch.setattr(doc_module, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    return calls


def _segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def _document(*segments: Segment) -> TranscriptDocument:
    return TranscriptDocument(
        video_id=VIDEO_ID,
        language="en",
        language_label="English",
        segments=list(segments),
        source=SourceInfo(kind=SourceKind.MANUAL_CAPTIONS, backend="youtube-transcript-api"),
    )


def _metadata(**overrides) -> VideoMetadata:
    values: dict = {
        "video_id": VIDEO_ID,
        "title": "Great Talk",
        "channel": "Great Channel",
        "upload_date": "20240131",
        "duration_sec": 120.0,
        "description": "About this talk.",
        "webpage_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "thumbnail": "https://example.com/t.jpg",
    }
    values.update(overrides)
    return VideoMetadata(**values)


def _frontmatter_lines(md: str) -> list[str]:
    lines = md.splitlines()
    end = lines[1:].index("---") + 1
    return lines[1:end]


class TestFetchVideoMetadata:
    def test_maps_fields_and_chapters(self, monkeypatch):
        calls = install_fake_ydl(monkeypatch, info=dict(CANNED_INFO))

        meta = fetch_video_metadata(VIDEO_ID)

        assert meta == VideoMetadata(
            video_id=VIDEO_ID,
            title="Sample: Talk",
            uploader="Some Uploader",
            channel="Some Channel",
            upload_date="20240131",
            duration_sec=123.4,
            description="A description.",
            webpage_url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            thumbnail="https://example.com/thumb.jpg",
            chapters=[Chapter(title="Intro", start=0.0), Chapter(title="Main", start=90.5)],
        )
        assert calls[0]["url"] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
        assert calls[0]["download"] is False
        assert calls[0]["opts"]["quiet"] is True
        assert calls[0]["opts"]["noprogress"] is True
        assert calls[0]["opts"]["skip_download"] is True

    def test_missing_fields_default_to_none_and_empty(self, monkeypatch):
        install_fake_ydl(monkeypatch, info={"id": VIDEO_ID, "title": "Only Title"})

        meta = fetch_video_metadata(VIDEO_ID)

        assert meta.title == "Only Title"
        assert meta.uploader is None
        assert meta.channel is None
        assert meta.upload_date is None
        assert meta.duration_sec is None
        assert meta.chapters == []

    def test_download_error_becomes_video_unavailable(self, monkeypatch):
        error = DownloadError("video unavailable")
        install_fake_ydl(monkeypatch, exc=error)

        with pytest.raises(VideoUnavailableError, match="video unavailable") as excinfo:
            fetch_video_metadata(VIDEO_ID)

        assert excinfo.value.__cause__ is error

    def test_generic_error_becomes_retryable_backend_error(self, monkeypatch):
        install_fake_ydl(monkeypatch, exc=RuntimeError("boom"))

        with pytest.raises(BackendError) as excinfo:
            fetch_video_metadata(VIDEO_ID)

        assert excinfo.value.backend == "ytdlp_metadata"
        assert excinfo.value.retryable is True
        assert "RuntimeError: boom" in str(excinfo.value)


class TestFrontmatter:
    def test_quotes_hostile_title_exactly(self):
        title = 'Weird: "quotes"\nand newline'
        md = compose_markdown_doc(_metadata(title=title), _document(_segment(0, 2, "hi")))

        expected_line = f"title: {json.dumps(title)}"
        frontmatter = _frontmatter_lines(md)
        assert expected_line in frontmatter
        assert len(frontmatter) == len({line.split(":")[0] for line in frontmatter})

    def test_key_order_and_values(self):
        md = compose_markdown_doc(_metadata(), _document(_segment(0, 2, "hi")))

        frontmatter = _frontmatter_lines(md)
        keys = [line.split(":")[0] for line in frontmatter]
        assert keys == [
            "video_id",
            "title",
            "url",
            "channel",
            "published",
            "duration_sec",
            "language",
            "source",
        ]
        assert 'video_id: "dQw4w9WgXcQ"' in frontmatter
        assert 'channel: "Great Channel"' in frontmatter
        assert 'published: "2024-01-31"' in frontmatter
        assert "duration_sec: 120.0" in frontmatter
        assert 'language: "en"' in frontmatter
        assert 'source: "manual_captions/youtube-transcript-api"' in frontmatter

    def test_canonical_url_fallback_when_webpage_url_missing(self):
        md = compose_markdown_doc(_metadata(webpage_url=None), _document(_segment(0, 2, "hi")))

        assert f'url: "https://www.youtube.com/watch?v={VIDEO_ID}"' in md

    def test_unparseable_upload_date_omits_published(self):
        md = compose_markdown_doc(_metadata(upload_date="junk"), _document(_segment(0, 2, "hi")))

        assert not any(line.startswith("published:") for line in _frontmatter_lines(md))


class TestDescription:
    def test_long_description_truncated_with_marker(self):
        long_desc = "word " * 200
        md = compose_markdown_doc(_metadata(description=long_desc), _document(_segment(0, 2, "hi")))

        assert "## Description" in md
        assert "… (truncated)" in md
        description_body = md.split("## Description\n\n", 1)[1].split("\n\n## ", 1)[0]
        assert len(description_body) < 700

    def test_short_description_rendered_verbatim(self):
        md = compose_markdown_doc(_metadata(description="Short."), _document(_segment(0, 2, "hi")))

        assert "## Description\n\nShort.\n\n" in md
        assert "(truncated)" not in md

    def test_empty_description_omits_section(self):
        md = compose_markdown_doc(_metadata(description=None), _document(_segment(0, 2, "hi")))

        assert "## Description" not in md


class TestChaptersSection:
    def test_renders_bullet_list_with_short_timestamps(self):
        chapters = [Chapter(title="Intro", start=0.0), Chapter(title="Main", start=95.0)]
        md = compose_markdown_doc(_metadata(chapters=chapters), _document(_segment(0, 2, "hi")))

        assert "- **[0:00]** Intro" in md
        assert "- **[1:35]** Main" in md

    def test_no_chapters_omits_section_and_headings(self):
        md = compose_markdown_doc(_metadata(chapters=[]), _document(_segment(0, 2, "hi")))

        assert "## Chapters" not in md
        assert "### " not in md


GROUPED_SEGMENTS = (
    _segment(0.0, 2.0, "intro words"),
    _segment(11.0, 13.0, "chapter one text"),
    _segment(14.0, 16.0, "more chapter one"),
    _segment(21.0, 24.0, "chapter two text"),
)

COVERING_CHAPTERS = [
    Chapter(title="Intro Later", start=10.0),
    Chapter(title="Deep Dive", start=20.0),
]


class TestTranscriptBody:
    def test_grouped_under_headings_with_leading_paragraph_first(self):
        md = compose_markdown_doc(
            _metadata(chapters=COVERING_CHAPTERS), _document(*GROUPED_SEGMENTS)
        )
        expected = (
            "## Transcript\n\n"
            "**[0:00]** intro words\n\n"
            "### Intro Later\n\n"
            "**[0:11]** chapter one text more chapter one\n\n"
            "### Deep Dive\n\n"
            "**[0:21]** chapter two text\n"
        )

        assert md.endswith(expected)

    def test_chapters_not_covering_timeline_fall_back_to_flat(self):
        beyond = [Chapter(title="Beyond", start=999.0)]
        md = compose_markdown_doc(_metadata(chapters=beyond), _document(*GROUPED_SEGMENTS))

        assert "### Beyond" not in md
        assert "**[0:00]** intro words" in md

    def test_hour_format_anchor(self):
        chapters = [Chapter(title="All", start=0.0)]
        md = compose_markdown_doc(
            _metadata(chapters=chapters),
            _document(_segment(3605.0, 3607.0, "late words")),
        )

        assert "**[1:00:05]** late words" in md

    def test_paragraph_splits_on_gap_over_two_seconds(self):
        md = compose_markdown_doc(
            _metadata(),
            _document(_segment(0.0, 2.0, "alpha"), _segment(5.0, 7.0, "beta")),
        )

        assert "**[0:00]** alpha\n\n**[0:05]** beta" in md

    def test_internal_newlines_collapsed_within_paragraph(self):
        md = compose_markdown_doc(
            _metadata(),
            _document(_segment(0.0, 1.0, "line one"), _segment(1.0, 2.0, "line\ntwo")),
        )

        assert "**[0:00]** line one line two" in md

    def test_ends_with_exactly_one_trailing_newline(self):
        md = compose_markdown_doc(
            _metadata(chapters=COVERING_CHAPTERS), _document(*GROUPED_SEGMENTS)
        )

        assert md.endswith("\n")
        assert not md.endswith("\n\n")


def _doc(segment_count: int = 40, step: float = 30.0) -> TranscriptDocument:
    segments = [
        Segment(
            start=i * step,
            end=(i + 1) * step,
            text=f"Sentence {i} about drawdown and position sizing in some detail.",
        )
        for i in range(segment_count)
    ]
    return TranscriptDocument(
        video_id=VIDEO_ID,
        language="en",
        language_label="English",
        segments=segments,
        source=SourceInfo(kind=SourceKind.MANUAL_CAPTIONS, backend="captions_api"),
    )


class TestComposeIndex:
    def test_reports_the_path_and_the_cost_of_reading_in_full(self) -> None:
        transcript = _doc()
        index = compose_index(
            VideoMetadata(video_id=VIDEO_ID, title="A Talk", channel="Ch"),
            transcript,
            path="notes/x.md",
        )
        assert "file:     notes/x.md" in index
        assert "A Talk" in index
        assert "40 segments" in index
        assert "tokens if read in full" in index
        assert "grep -n" in index

    def test_is_far_smaller_than_the_document(self) -> None:
        transcript = _doc(segment_count=400)
        metadata = VideoMetadata(video_id=VIDEO_ID, title="A Talk")
        index = compose_index(metadata, transcript, path="x.md")
        document = compose_markdown_doc(metadata, transcript)
        assert len(index) < len(document) / 10

    def test_uses_chapters_when_present(self) -> None:
        metadata = VideoMetadata(
            video_id=VIDEO_ID,
            title="A Talk",
            chapters=[Chapter(title="Intro", start=0.0), Chapter(title="Backtests", start=300.0)],
        )
        index = compose_index(metadata, _doc(), path="x.md")
        assert "chapters:" in index
        assert "Intro" in index and "Backtests" in index

    def test_falls_back_to_a_time_bucketed_outline(self) -> None:
        index = compose_index(
            VideoMetadata(video_id=VIDEO_ID, title="A Talk"), _doc(), path="x.md"
        )
        assert "outline:" in index
        # One entry per bucket, not one per paragraph.
        entries = [line for line in index.splitlines() if line.startswith("  ") and "…" in line]
        assert 2 <= len(entries) <= 12

    def test_includes_cleanup_notes(self) -> None:
        index = compose_index(
            VideoMetadata(video_id=VIDEO_ID, title="A Talk"),
            _doc(),
            path="x.md",
            notes=["2 sponsor block(s) removed"],
        )
        assert "cleanup:  2 sponsor block(s) removed" in index

    def test_handles_an_empty_transcript(self) -> None:
        empty = _doc().model_copy(update={"segments": []})
        index = compose_index(VideoMetadata(video_id=VIDEO_ID, title="A Talk"), empty, path="x.md")
        assert "0 segments" in index
        assert "outline:" not in index
