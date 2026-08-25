"""SponsorBlock lookups for crowd-verified ad segments.

SponsorBlock is a public, keyless database of ad and self-promotion segments
submitted and voted on by viewers. Where a video is covered it is far more
accurate than any keyword heuristic; coverage is the catch, so this is a
supplement to local detection rather than a replacement for it.

Queries use the hash-prefix endpoint: only the first four hex characters of the
video id's SHA-256 land in the request, so the server learns that someone asked
about one of a few dozen videos and never which one.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["DEFAULT_CATEGORIES", "fetch_ranges"]

_API = "https://sponsor.ajay.app/api/skipSegments"
_HASH_PREFIX_LEN = 4
_DEFAULT_TIMEOUT = 8.0

# Only categories that are unambiguously not content. `filler`, `preview` and
# `music_offtopic` are left out: they cover real speech a viewer may want.
DEFAULT_CATEGORIES = ("sponsor", "selfpromo", "interaction")


def fetch_ranges(
    video_id: str,
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[tuple[float, float, str]] | None:
    """Return ``(start, end, category)`` triples for a video.

    ``[]`` means the video is not in the database; ``None`` means the lookup
    itself failed and the caller should fall back to local detection only.
    """
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:_HASH_PREFIX_LEN]
    query = urllib.parse.urlencode({"categories": json.dumps(list(categories))})
    url = f"{_API}/{digest}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # 404 is the documented "nothing for this prefix" answer.
        return [] if exc.code == 404 else None
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, list):
        return None

    ranges: list[tuple[float, float, str]] = []
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("videoID") != video_id:
            continue
        for segment in entry.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            # `mute`, `poi` and `chapter` submissions are not skips.
            if segment.get("actionType") != "skip":
                continue
            # Negative scores mean the community rejected the submission.
            if (segment.get("votes") or 0) < 0:
                continue
            bounds = segment.get("segment")
            if not isinstance(bounds, list) or len(bounds) != 2:
                continue
            try:
                start, end = float(bounds[0]), float(bounds[1])
            except (TypeError, ValueError):
                continue
            if end > start:
                ranges.append((start, end, str(segment.get("category") or "sponsor")))
    return sorted(ranges)
