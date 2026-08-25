from __future__ import annotations

import pytest

from ytx_core.errors import InvalidInputError, PlaylistNotSupportedError
from ytx_core.resolver import extract_video_id, looks_like_playlist_only

VALID_CASES = [
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("  dQw4w9WgXcQ\n", "dQw4w9WgXcQ"),
    ("a-b_cdefghi", "a-b_cdefghi"),
    ("_-0-_ZyxwvU", "_-0-_ZyxwvU"),
    (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&t=42s",
        "dQw4w9WgXcQ",
    ),
    ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&playlist=PL123", "dQw4w9WgXcQ"),
]

INVALID_IDS = [
    "hello world",
    "",
    "dQw4w9WgX",
    "dQw4w9WgXcQdQw4w9WgXcQ",
    "dQw4w9WgXc!",
]

INVALID_URLS = [
    "https://example.com",
    "https://www.youtube.com/watch?v=notanid",
]


@pytest.mark.parametrize(("text", "expected"), VALID_CASES)
def test_valid_references(text: str, expected: str):
    assert extract_video_id(text) == expected


@pytest.mark.parametrize("text", INVALID_IDS)
def test_invalid_ids(text: str):
    with pytest.raises(InvalidInputError):
        extract_video_id(text)


@pytest.mark.parametrize("text", INVALID_URLS)
def test_invalid_urls(text: str):
    with pytest.raises(InvalidInputError):
        extract_video_id(text)


def test_query_param_playlist_only_raises_playlist_error():
    with pytest.raises(PlaylistNotSupportedError):
        extract_video_id("https://www.youtube.com/watch?playlist=PL123")


def test_playlist_path_url_raises_playlist_error():
    with pytest.raises(PlaylistNotSupportedError):
        extract_video_id("https://www.youtube.com/playlist?list=PL1234567890a")


def test_watch_with_playlist_param_still_returns_id():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&playlist=X") == (
        "dQw4w9WgXcQ"
    )


LOOKS_LIKE_PLAYLIST_CASES = [
    ("https://www.youtube.com/watch?playlist=PL123", True),
    ("youtube.com/watch?playlist=PL123", True),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx", False),
    ("dQw4w9WgXcQ", False),
    ("https://www.youtube.com/playlist?list=PLx", True),
]


@pytest.mark.parametrize(("url", "expected"), LOOKS_LIKE_PLAYLIST_CASES)
def test_looks_like_playlist_only(url: str, expected: bool):
    assert looks_like_playlist_only(url) is expected
