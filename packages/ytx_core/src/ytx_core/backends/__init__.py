from __future__ import annotations

import importlib.util
import os

from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.backends.captions_api import CaptionsApiBackend

__all__ = [
    "CaptionsApiBackend",
    "FetchRequest",
    "TranscriptBackend",
    "default_backends",
]


def default_backends() -> list[TranscriptBackend]:
    backends: list[TranscriptBackend] = [CaptionsApiBackend()]
    enabled = (os.environ.get("YTX_ENABLE_ASR") or "").strip().lower() in {"1", "true"}
    if enabled and importlib.util.find_spec("faster_whisper") is not None:
        from ytx_core.backends.asr_whisper import AsrWhisperBackend

        backends.append(AsrWhisperBackend())
    return backends
