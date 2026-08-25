from __future__ import annotations

from ytx_core.models import TranscriptDocument

EXPORT_FORMATS = ("json", "srt", "vtt", "txt", "md")

_SENTENCE_GAP_SECONDS = 2.0
_MAX_SEGMENTS_PER_PARAGRAPH = 12
_MAX_WORDS_PER_PARAGRAPH = 250


def _timestamp(seconds: float, decimal_separator: str) -> str:
    clamped = max(seconds, 0.0)
    total_ms = round(clamped * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{millis:03d}"


def _clamped_blocks(doc: TranscriptDocument) -> list[tuple[float, float, str]]:
    blocks: list[tuple[float, float, str]] = []
    previous_start = 0.0
    for segment in doc.segments:
        start = max(segment.start, previous_start)
        end = max(segment.end, start + 0.001)
        blocks.append((start, end, segment.text))
        previous_start = start
    return blocks


def _paragraphs(doc: TranscriptDocument) -> list[list[tuple[float, float, str]]]:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    previous_end: float | None = None
    words_in_current = 0

    def flush() -> None:
        nonlocal current, words_in_current
        if current:
            groups.append(current)
        current = []
        words_in_current = 0

    for segment in doc.segments:
        if previous_end is not None and segment.start - previous_end > _SENTENCE_GAP_SECONDS:
            flush()
        if len(current) >= _MAX_SEGMENTS_PER_PARAGRAPH or (
            words_in_current >= _MAX_WORDS_PER_PARAGRAPH
        ):
            flush()
        text = segment.text.strip()
        current.append((segment.start, segment.end, segment.text))
        words_in_current += len(text.split())
        previous_end = segment.end
    flush()
    return groups


def _short_timestamp(seconds: float) -> str:
    total = max(int(seconds), 0)
    if total >= 3600:
        return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"
    return f"{total // 60}:{total % 60:02d}"


def to_json(doc: TranscriptDocument) -> str:
    return doc.model_dump_json(indent=2)


def to_srt(doc: TranscriptDocument) -> str:
    lines: list[str] = []
    for index, (start, end, text) in enumerate(_clamped_blocks(doc), start=1):
        lines.append(
            f"{index}\n{_timestamp(start, ',')} --> {_timestamp(end, ',')}\n{text.strip()}\n"
        )
    return "\n".join(lines)


def to_vtt(doc: TranscriptDocument) -> str:
    lines = ["WEBVTT"]
    for start, end, text in _clamped_blocks(doc):
        lines.append(f"\n{_timestamp(start, '.')} --> {_timestamp(end, '.')}\n{text.strip()}")
    return "\n".join(lines) + "\n"


def to_txt(doc: TranscriptDocument) -> str:
    paragraphs = [
        " ".join(text.strip() for _, _, text in group if text.strip()) for group in _paragraphs(doc)
    ]
    return "\n\n".join(p for p in paragraphs if p)


def to_md(doc: TranscriptDocument) -> str:
    header = (
        f"# Transcript {doc.video_id}\n\n"
        f"- language: {doc.language}\n"
        f"- source: {doc.source.kind.value} ({doc.source.backend})\n\n"
    )
    paragraphs = "\n\n".join(
        f"**[{_short_timestamp(group[0][0])}]** "
        + " ".join(text.strip() for _, _, text in group if text.strip())
        for group in _paragraphs(doc)
    )
    return header + paragraphs


_FORMATTERS = {
    "json": to_json,
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
    "md": to_md,
}


def format_transcript(doc: TranscriptDocument, fmt: str) -> str:
    formatter = _FORMATTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"unknown format {fmt!r} (expected one of {', '.join(EXPORT_FORMATS)})")
    return formatter(doc)
