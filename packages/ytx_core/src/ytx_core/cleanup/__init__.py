"""Render-time cleanup passes for AI-ready documents.

Run *after* the cache, so the cache always holds the raw transcript and
disabling cleanup never costs a re-fetch.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ytx_core.cleanup.lexicon import (
    GENERAL_PACK,
    Lexicon,
    LexiconPack,
    PackError,
    available_packs,
    build_lexicon,
    detect_packs,
    load_pack_file,
)
from ytx_core.cleanup.sponsorblock import DEFAULT_CATEGORIES, fetch_ranges
from ytx_core.cleanup.sponsors import (
    RemovedBlock,
    SponsorRange,
    detect_sponsor_ranges,
    remove_ranges,
    strip_description_links,
)
from ytx_core.models import TranscriptDocument

__all__ = [
    "DEFAULT_CATEGORIES",
    "GENERAL_PACK",
    "CleanupOptions",
    "CleanupReport",
    "Lexicon",
    "LexiconPack",
    "PackError",
    "RemovedBlock",
    "SponsorRange",
    "available_packs",
    "build_lexicon",
    "clean",
    "detect_packs",
    "detect_sponsor_ranges",
    "fetch_ranges",
    "load_pack_file",
    "remove_ranges",
    "strip_description_links",
]

# How much text the domain detector reads before deciding.
_DETECT_SAMPLE_CHARS = 4000


class CleanupOptions(BaseModel):
    fix_terms: bool = True
    #: Explicit domain packs. ``None`` auto-detects from the video's own text.
    packs: tuple[str, ...] | None = None
    #: Path to a user-supplied TOML pack, applied ahead of the bundled ones.
    custom_pack: str | None = None
    strip_sponsors: bool = True
    use_sponsorblock: bool = True
    clean_description: bool = True

    @classmethod
    def disabled(cls) -> CleanupOptions:
        return cls(
            fix_terms=False,
            strip_sponsors=False,
            use_sponsorblock=False,
            clean_description=False,
        )

    @property
    def any_enabled(self) -> bool:
        return (
            self.fix_terms
            or self.strip_sponsors
            or self.use_sponsorblock
            or self.clean_description
        )


class CleanupReport(BaseModel):
    lexicon_packs: list[str] = Field(default_factory=list)
    terms_fixed: int = 0
    removed_blocks: list[RemovedBlock] = Field(default_factory=list)
    segments_removed: int = 0
    description_trimmed: bool = False
    #: "hit", "miss", "unavailable" or "disabled".
    sponsorblock: str = "disabled"

    @property
    def notes(self) -> list[str]:
        """Human-readable lines describing what cleanup changed."""
        lines: list[str] = []
        if self.terms_fixed:
            lines.append(
                f"vocabulary repaired in {self.terms_fixed} segments"
                f" ({', '.join(self.lexicon_packs)})"
            )
        if self.removed_blocks:
            spans = ", ".join(
                f"{int(block.start // 60)}:{int(block.start % 60):02d}"
                f" [{'/'.join(block.sources)}]"
                for block in self.removed_blocks
            )
            lines.append(
                f"{len(self.removed_blocks)} sponsor block(s) removed"
                f" ({self.segments_removed} segments, at {spans})"
            )
        if self.sponsorblock == "unavailable":
            lines.append("SponsorBlock unreachable — local detection only")
        if self.description_trimmed:
            lines.append("affiliate links stripped from description")
        return lines


def _resolve_lexicon(
    options: CleanupOptions, detect_text: str
) -> tuple[Lexicon, list[str]]:
    names = options.packs if options.packs is not None else detect_packs(detect_text)
    extra: list[LexiconPack] = []
    if options.custom_pack:
        extra.append(load_pack_file(options.custom_pack))
    lexicon = build_lexicon(names, extra=extra)
    return lexicon, list(lexicon.names)


def _sponsor_ranges(
    doc: TranscriptDocument, video_id: str, options: CleanupOptions
) -> tuple[list[SponsorRange], str]:
    ranges: list[SponsorRange] = []
    status = "disabled"
    if options.use_sponsorblock:
        fetched = fetch_ranges(video_id)
        if fetched is None:
            status = "unavailable"
        else:
            status = "hit" if fetched else "miss"
            ranges.extend(
                SponsorRange(
                    start=start, end=end, sources=["sponsorblock"], categories=[category]
                )
                for start, end, category in fetched
            )
    if options.strip_sponsors:
        ranges.extend(detect_sponsor_ranges(doc))
    return ranges, status


def clean(
    doc: TranscriptDocument,
    *,
    video_id: str | None = None,
    title: str = "",
    description: str | None = None,
    options: CleanupOptions | None = None,
) -> tuple[TranscriptDocument, str | None, CleanupReport]:
    """Apply the enabled cleanup passes to a transcript and its description.

    Returns the cleaned transcript, the cleaned description, and a report of
    everything that changed.
    """
    opts = options or CleanupOptions()
    report = CleanupReport()
    if not opts.any_enabled:
        return doc, description, report

    ranges, report.sponsorblock = _sponsor_ranges(doc, video_id or doc.video_id, opts)
    if ranges:
        doc, removed = remove_ranges(doc, ranges)
        report.removed_blocks = removed
        report.segments_removed = sum(block.segment_count for block in removed)

    cleaned_description = description
    if opts.clean_description and description:
        cleaned_description = strip_description_links(description)

    if opts.fix_terms:
        detect_text = " ".join(
            part for part in (title, description or "", doc.full_text[:_DETECT_SAMPLE_CHARS])
        )
        lexicon, report.lexicon_packs = _resolve_lexicon(opts, detect_text)
        doc, report.terms_fixed = lexicon.apply(doc)
        if cleaned_description:
            cleaned_description = lexicon.fix(cleaned_description)

    report.description_trimmed = bool(description) and cleaned_description != description
    return doc, cleaned_description, report
