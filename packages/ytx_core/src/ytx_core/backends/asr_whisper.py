from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ytx_core.backends.audio import fetch_audio
from ytx_core.backends.base import FetchRequest, TranscriptBackend
from ytx_core.errors import BackendError
from ytx_core.models import Segment, SourceInfo, SourceKind, TranscriptDocument

_NAME = "faster_whisper"
_DEFAULT_MODEL = "large-v3-turbo"
_DEFAULT_DEVICE = "auto"
_DEFAULT_COMPUTE = "auto"


class AsrWhisperBackend(TranscriptBackend):
    """ASR fallback backend that transcribes downloaded audio via faster-whisper."""

    name = _NAME
    supports_language_listing = False

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model or os.environ.get("YTX_ASR_MODEL", _DEFAULT_MODEL)
        self.device = device or os.environ.get("YTX_ASR_DEVICE", _DEFAULT_DEVICE)
        self.compute_type = compute_type or os.environ.get("YTX_ASR_COMPUTE", _DEFAULT_COMPUTE)
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "ytx-audio"
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise BackendError(
                        _NAME,
                        "ASR extra not installed (pip install 'ytx-core[asr]')",
                        retryable=False,
                    ) from exc
                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type
                )
        return self._model

    def fetch(self, request: FetchRequest) -> TranscriptDocument:
        model = self._get_model()
        url = f"https://www.youtube.com/watch?v={request.video_id}"
        audio = fetch_audio(url, self.cache_dir)
        try:
            segments_iter: Iterator[Any]
            info: Any
            segments_iter, info = model.transcribe(
                str(audio), vad_filter=True, word_timestamps=False
            )
            segments = [
                Segment(start=s.start, end=s.end, text=(s.text or "").strip())
                for s in segments_iter
            ]
            return TranscriptDocument(
                video_id=request.video_id,
                language=getattr(info, "language", None) or "unknown",
                language_label=None,
                is_generated=True,
                duration_sec=segments[-1].end if segments else 0.0,
                segments=segments,
                source=SourceInfo(
                    kind=SourceKind.ASR,
                    backend=_NAME,
                    model_version=self.model_name,
                ),
            )
        finally:
            audio.unlink(missing_ok=True)
