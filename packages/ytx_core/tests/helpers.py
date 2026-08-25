from __future__ import annotations

from datetime import UTC, datetime

from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument


def make_document(
    video_id: str = "dQw4w9WgXcQ",
    kind: SourceKind = SourceKind.AUTO_CAPTIONS,
    backend: str = "captions_api",
    segments: list[Segment] | None = None,
) -> TranscriptDocument:
    if segments is None:
        segments = [
            Segment(start=0.0, end=1.5, text="Hello world"),
            Segment(start=1.5, end=3.0, text="Second line"),
        ]
    return TranscriptDocument(
        video_id=video_id,
        language="en",
        language_label="English",
        is_generated=kind != SourceKind.MANUAL_CAPTIONS,
        duration_sec=max(segment.end for segment in segments),
        segments=segments,
        source=SourceInfo(kind=kind, backend=backend),
        fetched_at=datetime.now(UTC),
    )
