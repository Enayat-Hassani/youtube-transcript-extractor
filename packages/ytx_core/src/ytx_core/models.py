from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    MANUAL_CAPTIONS = "manual_captions"
    AUTO_CAPTIONS = "auto_captions"
    ASR = "asr"


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None


class SourceInfo(BaseModel):
    kind: SourceKind
    backend: str
    model_version: str | None = None


class LanguageOption(BaseModel):
    language_code: str
    language_label: str | None = None
    kind: SourceKind | None = None
    is_translatable: bool = False


class TranscriptDocument(BaseModel):
    video_id: str
    language: str
    language_label: str | None = None
    is_generated: bool = True
    duration_sec: float | None = None
    segments: list[Segment]
    source: SourceInfo
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def last_end(self) -> float:
        return max((s.end for s in self.segments), default=0.0)
