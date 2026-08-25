from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from ytx_core.errors import InvalidInputError, PlaylistNotSupportedError

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")

_HOSTS_WITH_PATH_ID = {"youtube.com", "m.youtube.com", "music.youtube.com"}
_PATH_PREFIXES = ("shorts", "embed", "live")


def _clean(text: str) -> str:
    return text.strip()


def _valid_id(candidate: str | None) -> str | None:
    if candidate and _VIDEO_ID_RE.fullmatch(candidate):
        return candidate
    return None


def _split_url(text: str) -> tuple[str, str, dict[str, list[str]]]:
    candidate = _clean(text)
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parts = urlsplit(candidate)
    host = (parts.netloc or "").lower().removeprefix("www.")
    return host, parts.path, parse_qs(parts.query)


def _find_video_id(host: str, path: str, params: dict[str, list[str]]) -> str | None:
    query_id = _valid_id((params.get("v") or [None])[0])
    if query_id:
        return query_id
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None
    if host == "youtu.be":
        return _valid_id(segments[0])
    if host in _HOSTS_WITH_PATH_ID and segments[0] in _PATH_PREFIXES and len(segments) >= 2:
        return _valid_id(segments[1])
    return None


def _is_playlist_path(path: str) -> bool:
    segments = [segment for segment in path.split("/") if segment]
    return bool(segments) and segments[0] == "playlist"


def extract_video_id(text: str) -> str:
    """Extract an 11-character YouTube video id from a raw id or a URL."""
    candidate = _clean(text)
    direct = _valid_id(candidate)
    if direct:
        return direct
    try:
        host, path, params = _split_url(text)
    except ValueError as exc:
        raise InvalidInputError(f"Could not parse {text!r} as a YouTube video reference") from exc
    video_id = _find_video_id(host, path, params)
    if video_id:
        return video_id
    if "playlist" in params or _is_playlist_path(path):
        raise PlaylistNotSupportedError(
            f"Playlist URLs are not supported yet (no video id in {text!r})"
        )
    raise InvalidInputError(f"Could not extract a YouTube video id from {text!r}")


def looks_like_playlist_only(url: str) -> bool:
    """Return True when the input references a playlist but contains no video id."""
    try:
        host, path, params = _split_url(url)
    except ValueError:
        return False
    if _valid_id(_clean(url)):
        return False
    playlist_ref = "playlist" in params or _is_playlist_path(path)
    return playlist_ref and _find_video_id(host, path, params) is None
