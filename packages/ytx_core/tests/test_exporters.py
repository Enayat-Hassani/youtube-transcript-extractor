from __future__ import annotations

import pytest

from ytx_core.exporters import EXPORT_FORMATS, format_transcript
from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument


def make_document() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="dQw4w9WgXcQ",
        language="en",
        language_label="English",
        is_generated=True,
        duration_sec=3602.0,
        segments=[
            Segment(start=0.0, end=1.5, text="Hello world"),
            Segment(start=1.5, end=1.4, text="Clamp me"),
            Segment(start=3600.0, end=3602.0, text="After the gap"),
        ],
        source=SourceInfo(kind=SourceKind.AUTO_CAPTIONS, backend="captions_api"),
    )


class TestJson:
    def test_round_trips_through_model(self):
        doc = make_document()
        rendered = format_transcript(doc, "json")
        assert TranscriptDocument.model_validate_json(rendered) == doc


class TestSrt:
    def test_exact_blocks_with_numbering_and_comma_millis(self):
        rendered = format_transcript(make_document(), "srt")
        assert rendered == (
            "1\n00:00:00,000 --> 00:00:01,500\nHello world\n"
            "\n2\n00:00:01,500 --> 00:00:01,501\nClamp me\n"
            "\n3\n01:00:00,000 --> 01:00:02,000\nAfter the gap\n"
        )

    def test_hours_grow_naturally(self):
        doc = make_document()
        doc.segments[2].start = 3661.5
        doc.segments[2].end = 3663.25
        rendered = format_transcript(doc, "srt")
        assert "01:01:01,500 --> 01:01:03,250" in rendered


class TestVtt:
    def test_header_and_dot_millis(self):
        rendered = format_transcript(make_document(), "vtt")
        lines = rendered.splitlines()
        assert lines[0] == "WEBVTT"
        assert "00:00:00.000 --> 00:00:01.500" in rendered
        assert "01:00:00.000 --> 01:00:02.000" in rendered
        assert "," not in "".join(lines[:2])


class TestTxt:
    def test_gap_over_two_seconds_splits_paragraphs(self):
        rendered = format_transcript(make_document(), "txt")
        paragraphs = rendered.split("\n\n")
        assert len(paragraphs) == 2
        assert paragraphs[0] == "Hello world Clamp me"
        assert paragraphs[1] == "After the gap"

    def test_small_gap_stays_one_paragraph(self):
        doc = make_document()
        doc.segments[1].end = 2.0
        doc.segments[2].start = 3.0
        doc.segments[2].end = 5.0
        rendered = format_transcript(doc, "txt")
        assert rendered == "Hello world Clamp me After the gap"


class TestMarkdown:
    def test_heading_metadata_and_paragraph_timestamps(self):
        rendered = format_transcript(make_document(), "md")
        head = (
            "# Transcript dQw4w9WgXcQ\n\n- language: en\n- source: auto_captions (captions_api)\n\n"
        )
        assert rendered.startswith(head)
        assert "**[0:00]** Hello world Clamp me" in rendered
        assert "**[1:00:00]** After the gap" in rendered


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        format_transcript(make_document(), "xyz")
    assert "unknown format" in str(excinfo.value)


def test_export_formats_constant():
    assert EXPORT_FORMATS == ("json", "srt", "vtt", "txt", "md")
