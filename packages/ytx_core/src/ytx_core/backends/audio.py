from __future__ import annotations

from pathlib import Path

import yt_dlp

from ytx_core.errors import BackendError

_BACKEND = "faster_whisper"


def fetch_audio(video_url: str, out_dir: Path) -> Path:
    """Download bestaudio for ``video_url`` into ``out_dir`` and return the file path."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with yt_dlp.YoutubeDL(
            {
                "format": "bestaudio[ext=m4a]/bestaudio",
                "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
                "quiet": True,
                "noprogress": True,
            }
        ) as ydl:
            info = ydl.extract_info(video_url, download=True)
        video_id = str(info["id"])
        matches = sorted(out_dir.glob(f"{video_id}.*"))
        if not matches:
            raise FileNotFoundError(f"no audio file produced for video {video_id}")
        return matches[-1]
    except Exception as exc:
        raise BackendError(_BACKEND, f"audio download failed: {exc}", retryable=True) from exc
