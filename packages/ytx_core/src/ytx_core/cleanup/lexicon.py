"""Vocabulary repair driven by pluggable domain packs.

Auto-generated captions mangle jargon by subject matter, so rules live in TOML
packs under ``packs/`` — one per domain — and a document is repaired with the
``general`` pack plus whichever domain packs its own text selects. Adding a
domain means dropping in a TOML file; no code change needed.

Packs apply domain-first, so a domain rule outranks a general one. Within a
pack, declaration order decides: rules compile into one alternation and Python
``|`` takes the first alternative matching at a position, so ``back testing``
must precede ``back test``. A single pass also means a replacement can never
re-trigger another rule.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ytx_core.errors import LexiconPackError
from ytx_core.models import Segment, TranscriptDocument

__all__ = [
    "GENERAL_PACK",
    "PackError",
    "Lexicon",
    "LexiconPack",
    "available_packs",
    "build_lexicon",
    "detect_packs",
    "load_pack_file",
]

GENERAL_PACK = "general"
# A pack needs this many distinct keyword hits before it is auto-selected, and
# at most this many domain packs are ever stacked onto `general`.
_MIN_DETECT_HITS = 2
_MAX_AUTO_PACKS = 2
# A secondary pack must be at least this strong relative to the leader. A
# trading video mentioning "code" and "software" should not pull in the tech
# pack unless it is genuinely about both.
_SECONDARY_RATIO = 0.5


@dataclass(frozen=True)
class LexiconPack:
    name: str
    description: str
    detect: tuple[str, ...]
    rules: tuple[tuple[str, str], ...]


#: Raised for an unknown, missing or malformed pack. Part of the shared error
#: hierarchy so callers can report it like any other bad input.
PackError = LexiconPackError


def _parse_pack(data: Mapping[str, object], origin: str) -> LexiconPack:
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise PackError(f"{origin}: missing a string 'name'")
    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        raise PackError(f"{origin}: 'rules' must be an array")
    rules: list[tuple[str, str]] = []
    for index, entry in enumerate(raw_rules):
        if not isinstance(entry, list) or len(entry) != 2:
            raise PackError(f"{origin}: rule {index} must be [pattern, replacement]")
        pattern, replacement = entry
        if not isinstance(pattern, str) or not isinstance(replacement, str):
            raise PackError(f"{origin}: rule {index} must hold two strings")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise PackError(f"{origin}: rule {index} has an invalid pattern: {exc}") from exc
        rules.append((pattern, replacement))
    detect = tuple(
        str(word).lower() for word in (data.get("detect") or []) if isinstance(word, str)
    )
    return LexiconPack(
        name=name,
        description=str(data.get("description") or ""),
        detect=detect,
        rules=tuple(rules),
    )


def _packs_dir() -> Path:
    return Path(__file__).parent / "packs"


@lru_cache(maxsize=1)
def available_packs() -> Mapping[str, LexiconPack]:
    """Every pack bundled under ``packs/``, keyed by name."""
    packs: dict[str, LexiconPack] = {}
    for path in sorted(_packs_dir().glob("*.toml")):
        pack = _parse_pack(tomllib.loads(path.read_text(encoding="utf-8")), path.name)
        packs[pack.name] = pack
    return packs


def load_pack_file(path: str | Path) -> LexiconPack:
    """Load a user-supplied pack from an arbitrary path."""
    target = Path(path)
    if not target.is_file():
        raise PackError(f"lexicon pack not found: {target}")
    return _parse_pack(tomllib.loads(target.read_text(encoding="utf-8")), target.name)


def detect_packs(
    text: str, *, limit: int = _MAX_AUTO_PACKS, min_hits: int = _MIN_DETECT_HITS
) -> tuple[str, ...]:
    """Pick domain packs whose keywords appear in ``text``, strongest first."""
    lowered = text.lower()
    scored: list[tuple[int, str]] = []
    for name, pack in available_packs().items():
        if name == GENERAL_PACK or not pack.detect:
            continue
        hits = sum(1 for word in pack.detect if word in lowered)
        if hits >= min_hits:
            scored.append((hits, name))
    if not scored:
        return ()
    scored.sort(key=lambda item: (-item[0], item[1]))
    leader = scored[0][0]
    strong = [
        name for hits, name in scored[:limit] if hits >= leader * _SECONDARY_RATIO
    ]
    return tuple(strong)


class Lexicon:
    """A compiled set of packs, applied in a single pass."""

    def __init__(self, packs: Sequence[LexiconPack]) -> None:
        self.packs = tuple(packs)
        flattened = [rule for pack in self.packs for rule in pack.rules]
        self._replacements = {
            f"g{index}": replacement for index, (_, replacement) in enumerate(flattened)
        }
        self._regex = (
            re.compile(
                "|".join(
                    f"(?P<g{index}>{pattern})" for index, (pattern, _) in enumerate(flattened)
                ),
                re.IGNORECASE,
            )
            if flattened
            else None
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(pack.name for pack in self.packs)

    @property
    def rule_count(self) -> int:
        return len(self._replacements)

    def _substitute(self, match: re.Match[str]) -> str:
        replacement = self._replacements.get(match.lastgroup or "")
        if replacement is None:
            return match.group(0)
        matched = match.group(0)
        # Keep a sentence-initial capital when the canonical form is lowercase.
        if matched[:1].isupper() and replacement[:1].islower():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    def fix(self, text: str) -> str:
        if self._regex is None:
            return text
        return self._regex.sub(self._substitute, text)

    def apply(self, doc: TranscriptDocument) -> tuple[TranscriptDocument, int]:
        """Return a copy of ``doc`` with repaired vocabulary, plus segments changed."""
        if self._regex is None:
            return doc, 0
        changed = 0
        segments: list[Segment] = []
        for segment in doc.segments:
            fixed = self.fix(segment.text)
            if fixed != segment.text:
                changed += 1
                segments.append(segment.model_copy(update={"text": fixed}))
            else:
                segments.append(segment)
        if changed == 0:
            return doc, 0
        return doc.model_copy(update={"segments": segments}), changed


def _ordered(names: Iterable[str]) -> tuple[str, ...]:
    """Domain packs first, then ``general``, so a domain rule wins."""
    unique = list(dict.fromkeys(names))
    domains = [name for name in unique if name != GENERAL_PACK]
    return (*domains, GENERAL_PACK)


@lru_cache(maxsize=32)
def _build_cached(names: tuple[str, ...]) -> Lexicon:
    packs = available_packs()
    missing = [name for name in names if name not in packs]
    if missing:
        known = ", ".join(sorted(packs))
        raise PackError(f"unknown lexicon pack(s): {', '.join(missing)} (available: {known})")
    return Lexicon([packs[name] for name in names])


def build_lexicon(
    names: Iterable[str] = (), *, extra: Sequence[LexiconPack] = ()
) -> Lexicon:
    """Compile ``general`` plus the named domain packs, plus any custom packs.

    Custom packs outrank bundled ones, matching the domain-first ordering.
    """
    cached = _build_cached(_ordered(names))
    if not extra:
        return cached
    return Lexicon([*extra, *cached.packs])
