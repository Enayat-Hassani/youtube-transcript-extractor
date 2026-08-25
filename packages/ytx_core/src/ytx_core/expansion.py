from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

import yt_dlp

from ytx_core.errors import InvalidInputError

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")

_CHANNEL_SEGMENTS = {"channel", "c", "user"}


def _valid_id(candidate: object) -> str | None:
    if isinstance(candidate, str) and _VIDEO_ID_RE.fullmatch(candidate):
        return candidate
    return None


def _split_url(text: str) -> tuple[str, str, dict[str, list[str]]]:
    stripped = text.strip()
    if "://" not in stripped:
        stripped = f"https://{stripped}"
    parts = urlsplit(stripped)
    host = (parts.netloc or "").lower().removeprefix("www.")
    return host, parts.path, parse_qs(parts.query)


def is_expandable_input(text: str) -> bool:
    """Best-effort check for playlist/channel/handle references that need expansion."""
    candidate = text.strip()
    if _valid_id(candidate):
        return False
    lowered = candidate.lower()
    if lowered.startswith("@") or "/@" in lowered:
        return True
    try:
        _, path, params = _split_url(text)
    except ValueError:
        return "list=" in lowered
    if _valid_id((params.get("v") or [None])[0]):
        return False
    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[0] == "playlist":
        return True
    if segments and segments[0] in _CHANNEL_SEGMENTS and len(segments) >= 2:
        return True
    return "list" in params or "list=" in lowered


def expand_to_video_ids(url_or_id: str, *, limit: int = 500) -> list[str]:
    """Expand a video/playlist/channel/handle reference to YouTube video ids via yt-dlp."""
    options = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "noprogress": True,
        "skip_download": True,
    }
    try:
        ydl = yt_dlp.YoutubeDL(options)
        info = ydl.extract_info(url_or_id, download=False)
    except Exception as exc:
        raise InvalidInputError(f"could not expand {url_or_id!r}: {type(exc).__name__}") from exc

    entries = (info or {}).get("entries")
    if entries is not None:
        ids: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            video_id = _valid_id((entry or {}).get("id"))
            if video_id and video_id not in seen:
                seen.add(video_id)
                ids.append(video_id)
                if len(ids) >= limit:
                    break
        if not ids:
            raise InvalidInputError(f"no videos found at {url_or_id!r}")
        return ids

    single = _valid_id((info or {}).get("id"))
    if single:
        return [single]
    raise InvalidInputError(f"no videos found at {url_or_id!r}")
