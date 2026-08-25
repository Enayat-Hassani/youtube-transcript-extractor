from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument


def make_segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def make_document(
    video_id: str = "dQw4w9WgXcQ",
    language: str = "en",
    kind: SourceKind = SourceKind.AUTO_CAPTIONS,
    backend: str = "captions_api",
    is_generated: bool = True,
    fetched_at: datetime | None = None,
    segments: list[Segment] | None = None,
) -> TranscriptDocument:
    if segments is None:
        segments = [
            make_segment(0.0, 1.5, "Hello world"),
            make_segment(1.5, 3.0, "Second line"),
        ]
    return TranscriptDocument(
        video_id=video_id,
        language=language,
        language_label="English",
        is_generated=is_generated,
        duration_sec=max(s.end for s in segments),
        segments=segments,
        source=SourceInfo(kind=kind, backend=backend),
        fetched_at=fetched_at or datetime.now(UTC),
    )


def old_timestamp(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


@pytest.fixture
def document() -> TranscriptDocument:
    return make_document()
