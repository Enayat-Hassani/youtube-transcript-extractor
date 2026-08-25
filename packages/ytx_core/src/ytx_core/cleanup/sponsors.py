"""Local detection and removal of sponsor / self-promotion blocks.

Detection produces time ranges, so crowd-sourced ranges from SponsorBlock and
locally detected ones merge into a single removal pass. Ads are detected as
*blocks* rather than single segments, since an ad read runs tens of seconds
and only a few of its segments carry a keyword.

Local detection is deliberately conservative — it prefers leaving an ad in to
removing content: a block needs two distinct signal categories (or one strong
one: a discount code or named sponsor), bounds expand only over segments
naming the advertiser, and removal is capped both per block and per transcript.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

from ytx_core.models import TranscriptDocument

__all__ = [
    "RemovedBlock",
    "SponsorRange",
    "detect_sponsor_ranges",
    "remove_ranges",
    "strip_description_links",
]

# Non-hit segments tolerated between two hits before a block is split.
_MAX_GAP_SEGMENTS = 4
# A locally detected "ad" longer than this is almost certainly real content.
_MAX_BLOCK_SECONDS = 210.0
_MIN_HITS = 2
_MIN_CATEGORIES = 2
# Categories decisive enough to qualify a block unaided.
_STRONG_CATEGORIES = frozenset({"promo_code", "sponsor"})
# A token naming the advertiser: long enough to be distinctive, and present in
# no more than this fraction of the document's segments.
_BRAND_MIN_LENGTH = 4
_BRAND_MAX_DOC_FRACTION = 0.05
# The fraction degenerates on short transcripts; keep an absolute floor.
_BRAND_MIN_CEILING = 4
# Hard stop on runaway expansion, independent of _MAX_BLOCK_SECONDS.
_MAX_EXPAND_SEGMENTS = 20
# A single ad read is ~6-12 segments, which can exceed 25% of a short video.
# Allow that floor, but never strip more than half of any transcript.
_MAX_REMOVED_FRACTION = 0.25
_MIN_BUDGET_SEGMENTS = 12

_SIGNALS: dict[str, tuple[str, ...]] = {
    "promo_code": (
        r"\b\d{1,3}\s*%\s*off\b",
        r"\buse\s+(?:the\s+)?code\b",
        r"\bpromo\s+code\b",
        r"\bdiscount\s+code\b",
        r"\bcoupon\b",
        r"\bcode\s*:\s*\w+",
    ),
    "sponsor": (
        r"\bsponsored\s+by\b",
        r"\bour\s+sponsors?\b",
        r"\btoday'?s\s+sponsor\b",
        r"\bbrought\s+to\s+you\s+by\b",
        r"\bfor\s+sponsoring\b",
    ),
    "cta_link": (
        r"\blinks?\s+(?:for\s+that\s+|to\s+that\s+)?(?:are\s+|is\s+)?in\s+the\s+description\b",
        r"\bin\s+the\s+description\s+below\b",
        r"\bclick\s+the\s+link\b",
        r"\bcheck\s+out\s+the\s+link\b",
        r"\bfirst\s+link\b",
        r"\bsign\s+up\b",
    ),
    "channel_plug": (
        r"\bsubscribe\b",
        r"\bhit\s+the\s+like\b",
        r"\bsmash\s+(?:that|the)\s+like\b",
        r"\blike\s+button\b",
        r"\bring\s+the\s+bell\b",
        r"\bjoin\s+(?:our|the)\s+discord\b",
        r"\bdiscord\s+(?:community|server)\b",
        r"\bgiveaways?\b",
        r"\bfree\s+pdf\b",
        r"\bput\s+your\s+email\b",
        r"\bnewsletter\b",
    ),
    "offer": (
        r"\bfree\s+trial\b",
        r"\baffiliate\b",
        r"\bcompletely\s+for\s+free\b",
        r"\bfor\s+free,?\s+go\s+to\b",
        r"\bstarting\s+at\s+\$\d",
    ),
    "interrupt": (
        r"\bbefore\s+we\s+(?:get|dive|jump)\s+in(?:to)?\b",
        r"\bquick\s+(?:word|break|message)\b",
        r"\bbefore\s+we\s+continue\b",
        r"\bthank\s+you\s+for\s+all\s+of\s+your\s+support\b",
        r"\blet'?s\s+get\s+(?:in)?to\s+(?:this|the)\s+(?:episode|video)\b",
    ),
}

_COMPILED = {
    category: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for category, patterns in _SIGNALS.items()
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'&-]{2,}")
_URL_RE = re.compile(r"https?://\S+|\b[\w-]+\.(?:com|net|io|co|gg|me|link)\b", re.IGNORECASE)
_PROMO_LINE_RE = re.compile(
    r"\d{1,3}\s*%\s*off|\bcode\s*:|\buse\s+code\b|\bpromo\b|\baffiliate\b|\bfree\s+trial\b",
    re.IGNORECASE,
)
# Runs of box-drawing / punctuation used as visual separators.
_SEPARATOR_RE = re.compile(r"^[\W_]{4,}$")


class SponsorRange(BaseModel):
    """A stretch of the timeline believed to be an ad.

    ``sources`` is a list because overlapping ranges merge: a block both
    SponsorBlock and local detection flagged carries each of them.
    """

    start: float
    end: float
    sources: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Crowd-verified ranges are trusted past the local false-positive guards."""
        return "sponsorblock" in self.sources


class RemovedBlock(BaseModel):
    """A run of segments that was actually dropped."""

    start: float
    end: float
    segment_count: int
    sources: list[str]
    categories: list[str]


def _categories_for(text: str) -> set[str]:
    return {
        category
        for category, patterns in _COMPILED.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _candidate_blocks(doc: TranscriptDocument) -> list[tuple[list[int], set[str]]]:
    """Group hit segments into (hit_indices, categories) runs."""
    hits: list[tuple[int, set[str]]] = []
    for index, segment in enumerate(doc.segments):
        found = _categories_for(segment.text)
        if found:
            hits.append((index, found))
    if not hits:
        return []

    blocks: list[tuple[list[int], set[str]]] = []
    indices, categories = [hits[0][0]], set(hits[0][1])
    for index, found in hits[1:]:
        if index - indices[-1] - 1 <= _MAX_GAP_SEGMENTS:
            categories |= found
            indices.append(index)
            continue
        blocks.append((indices, categories))
        indices, categories = [index], set(found)
    blocks.append((indices, categories))
    return blocks


def _document_frequencies(doc: TranscriptDocument) -> Counter[str]:
    """How many segments each token appears in."""
    freq: Counter[str] = Counter()
    for segment in doc.segments:
        freq.update({token.lower() for token in _TOKEN_RE.findall(segment.text)})
    return freq


def _brand_tokens(
    doc: TranscriptDocument, freq: Counter[str], indices: list[int]
) -> frozenset[str]:
    """Distinctive tokens from the hit segments — the advertiser's name and its like."""
    ceiling = max(int(len(doc.segments) * _BRAND_MAX_DOC_FRACTION), _BRAND_MIN_CEILING)
    return frozenset(
        token.lower()
        for index in indices
        for token in _TOKEN_RE.findall(doc.segments[index].text)
        if len(token) >= _BRAND_MIN_LENGTH and freq[token.lower()] <= ceiling
    )


def _names_brand(text: str, brands: frozenset[str]) -> bool:
    return any(token.lower() in brands for token in _TOKEN_RE.findall(text))


def _expand(
    doc: TranscriptDocument, first: int, last: int, brands: frozenset[str]
) -> tuple[int, int]:
    """Widen a block over neighbouring segments that name the advertiser."""
    if not brands:
        return first, last
    segments = doc.segments
    start = first
    while (
        start > 0
        and first - start < _MAX_EXPAND_SEGMENTS
        and _names_brand(segments[start - 1].text, brands)
    ):
        start -= 1
    end = last
    while (
        end < len(segments) - 1
        and end - last < _MAX_EXPAND_SEGMENTS
        and _names_brand(segments[end + 1].text, brands)
    ):
        end += 1
    return start, end


def detect_sponsor_ranges(doc: TranscriptDocument) -> list[SponsorRange]:
    """Locate probable ad blocks from the transcript text alone."""
    candidates = _candidate_blocks(doc)
    if not candidates:
        return []
    freq = _document_frequencies(doc)

    ranges: list[SponsorRange] = []
    for indices, categories in candidates:
        if len(indices) < _MIN_HITS:
            continue
        if len(categories) < _MIN_CATEGORIES and not (categories & _STRONG_CATEGORIES):
            continue
        first, last = _expand(doc, indices[0], indices[-1], _brand_tokens(doc, freq, indices))
        start, end = doc.segments[first].start, doc.segments[last].end
        if end - start > _MAX_BLOCK_SECONDS:
            continue
        ranges.append(
            SponsorRange(
                start=start,
                end=end,
                sources=["heuristic"],
                categories=sorted(categories),
            )
        )
    return ranges


def _merge(ranges: list[SponsorRange]) -> list[SponsorRange]:
    """Collapse overlapping ranges, keeping every contributing source."""
    if not ranges:
        return []
    merged: list[SponsorRange] = []
    for current in sorted(ranges, key=lambda item: (item.start, item.end)):
        if merged and current.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = SponsorRange(
                start=previous.start,
                end=max(previous.end, current.end),
                sources=sorted(set(previous.sources) | set(current.sources)),
                categories=sorted(set(previous.categories) | set(current.categories)),
            )
            continue
        merged.append(current.model_copy())
    return merged


def _segments_in(doc: TranscriptDocument, span: SponsorRange) -> list[int]:
    """Indices whose midpoint falls inside the range."""
    return [
        index
        for index, segment in enumerate(doc.segments)
        if span.start <= (segment.start + segment.end) / 2 <= span.end
    ]


def remove_ranges(
    doc: TranscriptDocument, ranges: list[SponsorRange]
) -> tuple[TranscriptDocument, list[RemovedBlock]]:
    """Drop the segments covered by ``ranges``, subject to the removal cap."""
    total = len(doc.segments)
    if total == 0 or not ranges:
        return doc, []

    # Local guesses get the tight budget; verified ranges are bounded only by
    # the half-transcript ceiling. Both count against that ceiling.
    ceiling = total // 2
    local_budget = min(max(int(total * _MAX_REMOVED_FRACTION), _MIN_BUDGET_SEGMENTS), ceiling)
    resolved = [(span, _segments_in(doc, span)) for span in _merge(ranges)]
    # Verified ranges first, then the smallest local guesses.
    resolved.sort(key=lambda item: (not item[0].verified, len(item[1])))

    drop: set[int] = set()
    kept: list[tuple[SponsorRange, list[int]]] = []
    for span, indices in resolved:
        limit = ceiling if span.verified else local_budget
        if not indices or len(drop) + len(indices) > limit:
            continue
        drop.update(indices)
        kept.append((span, indices))

    if not drop:
        return doc, []

    removed = [
        RemovedBlock(
            start=doc.segments[indices[0]].start,
            end=doc.segments[indices[-1]].end,
            segment_count=len(indices),
            sources=list(span.sources),
            categories=span.categories,
        )
        for span, indices in sorted(kept, key=lambda item: item[1][0])
    ]
    segments = [segment for index, segment in enumerate(doc.segments) if index not in drop]
    return doc.model_copy(update={"segments": segments}), removed


def strip_description_links(text: str) -> str:
    """Drop affiliate/link-only lines and decorative separators from a description."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if _SEPARATOR_RE.match(stripped):
            continue
        has_url = _URL_RE.search(stripped) is not None
        # An offer pitched with a link is an ad however long the label is.
        if has_url and _PROMO_LINE_RE.search(stripped):
            continue
        without_urls = _URL_RE.sub("", stripped).strip(" \t-–—:|·,")
        # A line that is mostly URL, or a short label wrapped around one.
        if has_url and len(without_urls) <= 60:
            continue
        lines.append(line)
    out: list[str] = []
    for line in lines:
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()
