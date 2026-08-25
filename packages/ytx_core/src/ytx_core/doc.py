"""AI-ready Markdown document composition from video metadata and a transcript."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

import yt_dlp
from pydantic import BaseModel, Field
from yt_dlp.utils import DownloadError

from ytx_core.errors import BackendError, VideoUnavailableError
from ytx_core.exporters.formats import _paragraphs, _short_timestamp
from ytx_core.models import TranscriptDocument
from ytx_core.screen import ScreenCapture

_DESCRIPTION_LIMIT = 600

__all__ = ["Chapter", "VideoMetadata", "compose_markdown_doc", "fetch_video_metadata"]


class Chapter(BaseModel):
    title: str
    start: float


class VideoMetadata(BaseModel):
    video_id: str
    title: str
    uploader: str | None = None
    channel: str | None = None
    upload_date: str | None = None
    duration_sec: float | None = None
    description: str | None = None
    webpage_url: str | None = None
    thumbnail: str | None = None
    chapters: list[Chapter] = Field(default_factory=list)


def fetch_video_metadata(video_id: str) -> VideoMetadata:
    """Fetch video metadata (title, chapters, dates) via yt-dlp."""
    options = {"quiet": True, "noprogress": True, "skip_download": True}
    try:
        ydl = yt_dlp.YoutubeDL(options)
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except DownloadError as exc:
        raise VideoUnavailableError(str(exc)) from exc
    except Exception as exc:
        raise BackendError(
            "ytdlp_metadata", f"{type(exc).__name__}: {exc}", retryable=True
        ) from exc

    data = info or {}
    chapters = [
        Chapter(
            title=str(entry.get("title") or ""),
            start=float(entry.get("start_time") or 0.0),
        )
        for entry in data.get("chapters") or []
    ]
    return VideoMetadata(
        video_id=video_id,
        title=data.get("title") or "",
        uploader=data.get("uploader"),
        channel=data.get("channel"),
        upload_date=data.get("upload_date"),
        duration_sec=data.get("duration"),
        description=data.get("description"),
        webpage_url=data.get("webpage_url"),
        thumbnail=data.get("thumbnail"),
        chapters=chapters,
    )


def _yaml_line(key: str, value: Any) -> str:
    return f"{key}: {json.dumps(value)}"


def _published(upload_date: str | None) -> str | None:
    if upload_date and len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    return None


def _truncate_description(text: str) -> str:
    if len(text) <= _DESCRIPTION_LIMIT:
        return text
    return text[:_DESCRIPTION_LIMIT].rstrip() + "… (truncated)"


def _paragraph_lines(transcript: TranscriptDocument) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    for group in _paragraphs(transcript):
        joined = " ".join(raw.strip() for _, _, raw in group if raw.strip())
        collapsed = " ".join(joined.split())
        if collapsed:
            result.append((group[0][0], f"**[{_short_timestamp(group[0][0])}]** {collapsed}"))
    return result


def _covers_timeline(chapters: list[Chapter], last_end: float) -> bool:
    return bool(chapters) and chapters[-1].start <= last_end


def _chapter_groups(
    paragraphs: list[tuple[float, str]], chapters: list[Chapter]
) -> tuple[list[str], dict[int, list[str]]]:
    starts = [chapter.start for chapter in chapters]
    leading: list[str] = []
    grouped: dict[int, list[str]] = {}
    for start, text in paragraphs:
        index = bisect_right(starts, start) - 1
        if index < 0:
            leading.append(text)
        else:
            grouped.setdefault(index, []).append(text)
    return leading, grouped


def _screen_lines(captures: Sequence[ScreenCapture]) -> list[tuple[float, str]]:
    return [
        (
            capture.time,
            f"> **[{_short_timestamp(capture.time)}] on screen** {capture.text}",
        )
        for capture in captures
    ]


def _transcript_body(
    transcript: TranscriptDocument,
    chapters: list[Chapter],
    screen: Sequence[ScreenCapture] = (),
) -> str:
    paragraphs = sorted(
        _paragraph_lines(transcript) + _screen_lines(screen), key=lambda item: item[0]
    )
    blocks: list[str] = []
    if _covers_timeline(chapters, transcript.last_end):
        leading, grouped = _chapter_groups(paragraphs, chapters)
        if leading:
            blocks.append("\n\n".join(leading))
        for index, chapter in enumerate(chapters):
            texts = grouped.get(index)
            if not texts:
                continue
            blocks.append(f"### {chapter.title}\n\n" + "\n\n".join(texts))
    elif paragraphs:
        blocks.append("\n\n".join(text for _, text in paragraphs))
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def _frontmatter(metadata: VideoMetadata, transcript: TranscriptDocument) -> list[str]:
    url = metadata.webpage_url or f"https://www.youtube.com/watch?v={metadata.video_id}"
    lines = [
        _yaml_line("video_id", metadata.video_id),
        _yaml_line("title", metadata.title),
        _yaml_line("url", url),
    ]
    channel = metadata.channel or metadata.uploader
    if channel:
        lines.append(_yaml_line("channel", channel))
    published = _published(metadata.upload_date)
    if published:
        lines.append(_yaml_line("published", published))
    if metadata.duration_sec is not None:
        lines.append(_yaml_line("duration_sec", metadata.duration_sec))
    lines.append(_yaml_line("language", transcript.language))
    source = f"{transcript.source.kind.value}/{transcript.source.backend}"
    lines.append(_yaml_line("source", source))
    return lines


def _outline(
    transcript: TranscriptDocument, chapters: list[Chapter], buckets: int = 9
) -> list[str]:
    """A navigable map of the transcript: chapters, or time-bucketed openers."""
    if chapters:
        return [f"{_short_timestamp(c.start)}  {c.title}" for c in chapters]
    paragraphs = _paragraph_lines(transcript)
    if not paragraphs:
        return []
    span = transcript.last_end or paragraphs[-1][0] or 1.0
    step = span / buckets
    lines: list[str] = []
    next_mark = 0.0
    for start, text in paragraphs:
        if start < next_mark:
            continue
        words = " ".join(text.split(" ")[1:])  # drop the **[mm:ss]** prefix
        lines.append(f"{_short_timestamp(start)}  {' '.join(words.split()[:14])}…")
        next_mark = start + step
    return lines


def compose_index(
    metadata: VideoMetadata,
    transcript: TranscriptDocument,
    *,
    path: str,
    notes: list[str] | None = None,
    screen: Sequence[ScreenCapture] = (),
) -> str:
    """A compact map of a written document, for an agent to read instead of it.

    The point is token economy: the agent gets the shape of the video and the
    file path, then greps or reads only the part it needs.
    """
    words = len(transcript.full_text.split())
    chapters = sorted(metadata.chapters, key=lambda chapter: chapter.start)
    duration = _short_timestamp(metadata.duration_sec or transcript.last_end)
    lines = [
        f"file:     {path}",
        f"title:    {metadata.title}",
        f"channel:  {metadata.channel or metadata.uploader or 'unknown'}",
        f"length:   {duration}  ·  {len(transcript.segments)} segments"
        f"  ·  ~{words * 4 // 3:,} tokens if read in full",
        f"language: {transcript.language_label or transcript.language}"
        f"  ({transcript.source.kind.value})",
    ]
    for note in notes or []:
        lines.append(f"cleanup:  {note}")
    if screen:
        lines.append(
            f"screen:   {len(screen)} on-screen text captures inline in the document"
        )
    outline = _outline(transcript, chapters)
    if outline:
        label = "chapters" if chapters else "outline"
        lines.append(f"\n{label}:")
        lines.extend(f"  {entry}" for entry in outline)
    lines.append(
        "\nRead only what you need, e.g."
        f' grep -n "keyword" {path}  ·  sed -n \'120,160p\' {path}'
    )
    return "\n".join(lines) + "\n"


def compose_markdown_doc(
    metadata: VideoMetadata,
    transcript: TranscriptDocument,
    *,
    notes: list[str] | None = None,
    screen: Sequence[ScreenCapture] = (),
) -> str:
    """Render metadata + transcript as a Markdown document with YAML frontmatter.

    ``notes`` are rendered under the banner to record what cleanup changed.
    """
    label = transcript.language_label or transcript.language
    source = f"{transcript.source.kind.value}/{transcript.source.backend}"
    banner = (
        f"> Extracted by ytx · {label} · {len(transcript.segments)} segments"
        f" · source: {source}\n"
    )
    if notes:
        banner += ">\n" + "".join(f"> Cleanup: {note}\n" for note in notes)
    parts = [
        "---\n" + "\n".join(_frontmatter(metadata, transcript)) + "\n---\n\n",
        f"# {metadata.title}\n\n",
        banner + "\n",
    ]
    if metadata.description:
        parts.append(f"## Description\n\n{_truncate_description(metadata.description)}\n\n")
    chapters = sorted(metadata.chapters, key=lambda chapter: chapter.start)
    if chapters:
        items = [
            f"- **[{_short_timestamp(chapter.start)}]** {chapter.title}" for chapter in chapters
        ]
        parts.append("## Chapters\n\n" + "\n".join(items) + "\n\n")
    parts.append("## Transcript\n\n")
    parts.append(_transcript_body(transcript, chapters, screen))
    return "".join(parts)
