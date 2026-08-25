from __future__ import annotations

import sys
import types

import pytest

import ytx_core.backends.asr_whisper as asr_module
from ytx_core.backends import default_backends
from ytx_core.backends.asr_whisper import AsrWhisperBackend
from ytx_core.backends.base import FetchRequest
from ytx_core.errors import BackendError
from ytx_core.models import SourceKind

VIDEO_ID = "dQw4w9WgXcQ"


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    language = "en"


class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.transcribe_kwargs: dict | None = None
        type(self).instances.append(self)

    def transcribe(self, audio_path: str, **kwargs):
        self.transcribe_kwargs = {"audio_path": audio_path, **kwargs}
        segments = [FakeSegment(0.0, 1.5, " Hello"), FakeSegment(1.5, 3.5, " world ")]
        return iter(segments), FakeInfo()


def install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


@pytest.fixture()
def fake_model_cls(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    cls = install_fake_faster_whisper(monkeypatch)
    cls.instances.clear()
    return cls


@pytest.fixture()
def fake_fetch_audio(monkeypatch: pytest.MonkeyPatch, tmp_path):
    audio_file = tmp_path / "fake.m4a"
    audio_file.write_bytes(b"audio")

    def _fetch(video_url: str, out_dir) -> object:
        assert video_url == f"https://www.youtube.com/watch?v={VIDEO_ID}"
        return audio_file

    monkeypatch.setattr(asr_module, "fetch_audio", _fetch)
    return audio_file


class TestDocumentMapping:
    def test_document_fields_are_mapped(self, fake_model_cls, fake_fetch_audio):
        doc = AsrWhisperBackend().fetch(FetchRequest(VIDEO_ID))
        assert doc.video_id == VIDEO_ID
        assert doc.language == "en"
        assert doc.language_label is None
        assert doc.is_generated is True
        assert [(s.start, s.end) for s in doc.segments] == [(0.0, 1.5), (1.5, 3.5)]
        assert [s.text for s in doc.segments] == ["Hello", "world"]
        assert doc.duration_sec == 3.5
        assert doc.source.kind == SourceKind.ASR
        assert doc.source.backend == "faster_whisper"
        assert doc.source.model_version == fake_model_cls.instances[-1].init_args[0]

    def test_empty_transcription_yields_zero_duration_and_unknown_language(
        self, monkeypatch, fake_fetch_audio
    ):
        class EmptyModel:
            def transcribe(self, audio_path: str, **kwargs):
                return iter([]), types.SimpleNamespace(language=None)

        monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
        backend = AsrWhisperBackend()
        monkeypatch.setattr(backend, "_model", EmptyModel(), raising=False)
        doc = backend.fetch(FetchRequest(VIDEO_ID))
        assert doc.segments == []
        assert doc.duration_sec == 0.0
        assert doc.language == "unknown"

    def test_audio_file_is_removed_after_fetch(self, fake_model_cls, fake_fetch_audio):
        AsrWhisperBackend().fetch(FetchRequest(VIDEO_ID))
        assert not fake_fetch_audio.exists()


class TestModelInitAndFlags:
    def test_vad_and_word_timestamp_flags_are_passed(self, fake_model_cls, fake_fetch_audio):
        AsrWhisperBackend().fetch(FetchRequest(VIDEO_ID))
        model = fake_model_cls.instances[-1]
        assert model.transcribe_kwargs["vad_filter"] is True
        assert model.transcribe_kwargs["word_timestamps"] is False

    def test_lazy_init_passes_config_to_whisper_model(self, fake_model_cls, fake_fetch_audio):
        backend = AsrWhisperBackend(model="tiny", device="cpu", compute_type="int8")
        backend.fetch(FetchRequest(VIDEO_ID))
        model = fake_model_cls.instances[-1]
        assert model.init_args[0] == "tiny"
        assert model.init_kwargs["device"] == "cpu"
        assert model.init_kwargs["compute_type"] == "int8"

    def test_double_checked_locking_returns_cached_instance(self, fake_model_cls, fake_fetch_audio):
        backend = AsrWhisperBackend()
        first = backend._get_model()
        second = backend._get_model()
        assert first is second
        assert len(fake_model_cls.instances) == 1


class TestMissingExtra:
    def test_missing_extra_raises_non_retryable_before_audio_fetch(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)

        def _boom(*args, **kwargs):
            raise AssertionError("fetch_audio must not be called when extra is missing")

        monkeypatch.setattr(asr_module, "fetch_audio", _boom)
        with pytest.raises(BackendError) as excinfo:
            AsrWhisperBackend().fetch(FetchRequest(VIDEO_ID))
        assert excinfo.value.retryable is False
        assert excinfo.value.backend == "faster_whisper"


class TestDefaultBackends:
    def test_asr_excluded_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("YTX_ENABLE_ASR", raising=False)
        names = [b.name for b in default_backends()]
        assert names == ["captions_api"]

    def test_asr_included_when_env_set_and_spec_present(self, monkeypatch, fake_model_cls):
        import importlib.util

        monkeypatch.setenv("YTX_ENABLE_ASR", "true")
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: real_find_spec(name) if name != "faster_whisper" else object(),
        )
        names = [b.name for b in default_backends()]
        assert names == ["captions_api", "faster_whisper"]

    def test_asr_excluded_when_env_set_but_spec_absent(self, monkeypatch):
        import importlib.util

        monkeypatch.setenv("YTX_ENABLE_ASR", "1")
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: real_find_spec(name) if name != "faster_whisper" else None,
        )
        names = [b.name for b in default_backends()]
        assert names == ["captions_api"]


class TestFetchAudioErrorMapping:
    def test_download_failure_maps_to_retryable_backend_error(self, monkeypatch, tmp_path):
        import ytx_core.backends.audio as audio_module

        class RaisingYoutubeDL:
            def __init__(self, options) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                return None

            def extract_info(self, url: str, download: bool):
                raise RuntimeError("connection reset by peer")

        monkeypatch.setattr(audio_module.yt_dlp, "YoutubeDL", RaisingYoutubeDL)
        with pytest.raises(BackendError) as excinfo:
            audio_module.fetch_audio("https://www.youtube.com/watch?v=x", tmp_path)
        assert excinfo.value.retryable is True
        assert excinfo.value.backend == "faster_whisper"
        assert "audio download failed" in str(excinfo.value)
