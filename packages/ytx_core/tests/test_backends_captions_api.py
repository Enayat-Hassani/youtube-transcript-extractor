from __future__ import annotations

import pytest
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

import ytx_core.backends.captions_api as captions_api_module
from ytx_core.backends.base import FetchRequest
from ytx_core.backends.captions_api import CaptionsApiBackend
from ytx_core.errors import (
    BackendError,
    NoTranscriptFoundError,
    TranscriptsDisabledError,
    VideoUnavailableError,
)
from ytx_core.models import SourceKind


class FakeSnippet:
    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


class FakeFetchedTranscript:
    def __init__(self, language_code: str, language: str, is_generated: bool, snippets):
        self.language_code = language_code
        self.language = language
        self.is_generated = is_generated
        self.snippets = list(snippets)

    def __iter__(self):
        return iter(self.snippets)


class FakeTranscript:
    def __init__(
        self,
        language_code: str,
        *,
        language: str | None = None,
        is_generated: bool = False,
        is_translatable: bool = False,
        fetched: FakeFetchedTranscript | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.language_code = language_code
        self.language = language or language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable
        self.fetched = fetched
        self.fetch_error = fetch_error
        self.fetch_calls = 0

    def fetch(self, preserve_formatting: bool = False) -> FakeFetchedTranscript:
        self.fetch_calls += 1
        if self.fetch_error is not None:
            raise self.fetch_error
        assert self.fetched is not None
        return self.fetched


def fetched_for(transcript: FakeTranscript) -> FakeFetchedTranscript:
    if transcript.fetched is not None:
        return transcript.fetched
    return FakeFetchedTranscript(
        transcript.language_code,
        transcript.language,
        transcript.is_generated,
        [FakeSnippet("Hello", 0.0, 1.5), FakeSnippet(" world", 1.5, 2.0)],
    )


class FakeYouTubeTranscriptApi:
    listing: list[FakeTranscript] = []
    list_error: Exception | None = None

    def __init__(self) -> None:
        self.list_calls: list[str] = []

    def list(self, video_id: str) -> list[FakeTranscript]:
        self.list_calls.append(video_id)
        if type(self).list_error is not None:
            raise type(self).list_error
        for transcript in type(self).listing:
            transcript.fetched = fetched_for(transcript)
        return list(type(self).listing)


@pytest.fixture(autouse=True)
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeYouTubeTranscriptApi]:
    FakeYouTubeTranscriptApi.listing = []
    FakeYouTubeTranscriptApi.list_error = None
    monkeypatch.setattr(captions_api_module, "YouTubeTranscriptApi", FakeYouTubeTranscriptApi)
    return FakeYouTubeTranscriptApi


VIDEO_ID = "dQw4w9WgXcQ"


class TestSelectionWithLanguages:
    def test_requested_code_matches_exact_then_region_prefix(self, fake_api):
        en_us = FakeTranscript("en-US")
        de = FakeTranscript("de")
        fake_api.listing = [de, en_us]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID, languages=("en",)))
        assert doc.language == "en-US"
        assert en_us.fetch_calls == 1
        assert de.fetch_calls == 0

    def test_requested_codes_are_tried_in_priority_order(self, fake_api):
        en = FakeTranscript("en")
        de = FakeTranscript("de")
        fake_api.listing = [en, de]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID, languages=("de", "en")))
        assert doc.language == "de"

    def test_no_match_raises_no_transcript_found_with_requested(self, fake_api):
        fake_api.listing = [FakeTranscript("fr")]
        backend = CaptionsApiBackend()
        with pytest.raises(NoTranscriptFoundError) as excinfo:
            backend.fetch(FetchRequest(VIDEO_ID, languages=("en", "de")))
        assert excinfo.value.requested == ["en", "de"]
        assert excinfo.value.video_id == VIDEO_ID


class TestDefaultSelection:
    def test_prefers_manual_english(self, fake_api):
        manual_en = FakeTranscript("en", is_generated=False)
        auto_en = FakeTranscript("en", is_generated=True)
        fake_api.listing = [auto_en, manual_en]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID))
        assert doc.is_generated is False
        assert manual_en.fetch_calls == 1

    def test_falls_back_to_any_manual(self, fake_api):
        manual_fr = FakeTranscript("fr", is_generated=False)
        auto_en = FakeTranscript("en", is_generated=True)
        fake_api.listing = [auto_en, manual_fr]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID))
        assert doc.language == "fr"
        assert doc.is_generated is False

    def test_falls_back_to_auto_when_nothing_is_manual(self, fake_api):
        auto_de = FakeTranscript("de", is_generated=True)
        fake_api.listing = [auto_de]
        doc = CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))
        assert doc.is_generated is True


class TestDocumentMapping:
    def test_fields_are_mapped_from_fetched_transcript(self, fake_api):
        data = FakeFetchedTranscript(
            "en",
            "English",
            True,
            [FakeSnippet("Hello", 0.0, 1.5), FakeSnippet(" world", 1.5, 2.0)],
        )
        fake_api.listing = [FakeTranscript("en", is_generated=True, fetched=data)]
        doc = CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))
        assert doc.video_id == VIDEO_ID
        assert doc.language == "en"
        assert doc.language_label == "English"
        assert doc.is_generated is True
        assert [(s.start, s.end) for s in doc.segments] == [(0.0, 1.5), (1.5, 3.5)]
        assert [s.text for s in doc.segments] == ["Hello", " world"]
        assert doc.duration_sec == 3.5
        assert doc.source.kind == SourceKind.AUTO_CAPTIONS
        assert doc.source.backend == "captions_api"

    def test_manual_transcript_maps_to_manual_kind(self, fake_api):
        data = FakeFetchedTranscript("de", "German", False, [FakeSnippet("Hallo", 0.0, 1.0)])
        fake_api.listing = [FakeTranscript("de", is_generated=False, fetched=data)]
        doc = CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID, languages=("de",)))
        assert doc.is_generated is False
        assert doc.source.kind == SourceKind.MANUAL_CAPTIONS

    def test_empty_fetch_yields_zero_duration(self, fake_api):
        data = FakeFetchedTranscript("en", "English", False, [])
        fake_api.listing = [FakeTranscript("en", is_generated=False, fetched=data)]
        doc = CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))
        assert doc.segments == []
        assert doc.duration_sec == 0.0


class TestFetchExceptionMapping:
    def test_transcripts_disabled_maps_to_definitive_error(self, fake_api):
        fake_api.listing = [
            FakeTranscript("en", fetch_error=TranscriptsDisabled(VIDEO_ID)),
        ]
        with pytest.raises(TranscriptsDisabledError):
            CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))

    def test_video_unavailable_maps_to_definitive_error(self, fake_api):
        fake_api.list_error = VideoUnavailable(VIDEO_ID)
        with pytest.raises(VideoUnavailableError) as excinfo:
            CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))
        assert VIDEO_ID in str(excinfo.value)

    def test_generic_exception_becomes_retryable_backend_error(self, fake_api):
        fake_api.list_error = RuntimeError("socket hang up")
        with pytest.raises(BackendError) as excinfo:
            CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))
        assert excinfo.value.retryable is True
        assert excinfo.value.backend == "captions_api"
        assert "RuntimeError" in str(excinfo.value)

    def test_library_no_transcript_found_maps_to_definitive_error(self, fake_api):
        fake_api.list_error = NoTranscriptFound(VIDEO_ID, ["en"], None)
        with pytest.raises(NoTranscriptFoundError):
            CaptionsApiBackend().fetch(FetchRequest(VIDEO_ID))


class TestListTranscripts:
    def test_listing_maps_to_language_options(self, fake_api):
        fake_api.listing = [
            FakeTranscript("en", is_generated=False, is_translatable=False),
            FakeTranscript("en", is_generated=True, is_translatable=True),
        ]
        options = CaptionsApiBackend().list_transcripts(VIDEO_ID)
        assert len(options) == 2
        assert options[0].kind == SourceKind.MANUAL_CAPTIONS
        assert options[0].is_translatable is False
        assert options[0].language_label == "en"
        assert options[1].kind == SourceKind.AUTO_CAPTIONS
        assert options[1].is_translatable is True

    def test_definitive_listing_errors_are_translated(self, fake_api):
        fake_api.list_error = TranscriptsDisabled(VIDEO_ID)
        with pytest.raises(TranscriptsDisabledError):
            CaptionsApiBackend().list_transcripts(VIDEO_ID)

    def test_transient_listing_errors_are_wrapped(self, fake_api):
        fake_api.list_error = OSError("connection reset")
        with pytest.raises(BackendError) as excinfo:
            CaptionsApiBackend().list_transcripts(VIDEO_ID)
        assert excinfo.value.retryable is True


class TranslatableFakeTranscript(FakeTranscript):
    def __init__(self, *args, translated: FakeFetchedTranscript | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.translated = translated
        self.translate_calls: list[str] = []

    def translate(self, language_code: str) -> TranslatableFakeTranscript:
        self.translate_calls.append(language_code)
        clone = TranslatableFakeTranscript(
            language_code,
            language=f"translated:{language_code}",
            is_generated=self.is_generated,
            is_translatable=True,
            fetched=self.translated,
        )
        return clone


class TestRegionalEnglishDefault:
    def test_default_prefers_regional_english_manual_over_other_manuals(self, fake_api):
        manual_ar = FakeTranscript("ar")
        manual_en_gb = FakeTranscript("en-GB", language="English (UK)")
        auto_en = FakeTranscript("en", is_generated=True)
        fake_api.listing = [manual_ar, manual_en_gb, auto_en]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID))
        assert doc.language == "en-GB"
        assert doc.language_label == "English (UK)"
        assert manual_en_gb.fetch_calls == 1
        assert manual_ar.fetch_calls == 0


class TestTranslation:
    def test_translate_to_translates_chosen_track(self, fake_api):
        translated = FakeFetchedTranscript(
            "en", "English", False, [FakeSnippet("Hello there", 0.0, 1.0)]
        )
        track = TranslatableFakeTranscript("ar", is_translatable=True, translated=translated)
        fake_api.listing = [track]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID, translate_to="en"))
        assert doc.language == "en"
        assert doc.language_label == "English"
        assert doc.segments[0].text == "Hello there"
        assert track.translate_calls == ["en"]

    def test_same_base_language_skips_translation(self, fake_api):
        track = TranslatableFakeTranscript("en-GB", is_translatable=True)
        fake_api.listing = [track]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID, translate_to="en"))
        assert track.translate_calls == []
        assert doc.language == "en-GB"

    def test_non_translatable_track_falls_back_to_original(self, fake_api):
        plain = FakeTranscript("ar")
        fake_api.listing = [plain]
        backend = CaptionsApiBackend()
        doc = backend.fetch(FetchRequest(VIDEO_ID, translate_to="en"))
        assert doc.language == "ar"
