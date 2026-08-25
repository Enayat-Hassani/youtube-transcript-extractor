"""OCR backends, best-available first: Apple Vision on macOS (via a cached
Swift helper), tesseract elsewhere.

Measured on a real 720p trading-app frame, 72-item ground truth:

================================  =======  ============  =====
engine                             native  2x upscaled    time
================================  =======  ============  =====
tesseract ``--psm 6``             33%      70%           3.4s
tesseract ``--psm 11``            56%      75%           3.1s
Apple Vision                      95%      97%           0.7s
================================  =======  ============  =====

Hence: Vision first, sparse page segmentation for tesseract, upscaling at
frame-grab time, and ``usesLanguageCorrection`` off (it corrupts tickers,
and disabling it is roughly twice as fast). Both engines reconstruct visual
lines — Vision's per-region boxes sharing a baseline are merged left-to-right.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["OcrEngine", "available_engine", "ocr_lines"]

_TIMEOUT = 30.0
_COMPILE_TIMEOUT = 120.0
_SWIFT_SOURCE = Path(__file__).with_name("vision_ocr.swift")
# Tesseract's per-word confidence floor. Below this it is mostly noise; much
# above it the small text on a screen recording starts being discarded.
_MIN_CONFIDENCE = 45.0
# Sparse mode reads scattered UI text far better than the default block mode.
_TESSERACT_PSM = "11"
# Vision boxes on one baseline belong to one line; tolerance scales with height.
_BASELINE_TOLERANCE = 0.6
_MIN_BASELINE_TOLERANCE = 0.004

_NUMERIC_TOKEN = re.compile(r"^[$€£]?\d[\d.,:%/-]*[%a-zA-Z]?$")
_WORD_TOKEN = re.compile(r"^[A-Za-z][A-Za-z'./-]{2,}$")
_VOWEL = re.compile(r"[aeiouAEIOU]")


@dataclass(frozen=True)
class OcrEngine:
    name: str
    #: None when the engine is usable; otherwise why it is not.
    unavailable: str | None = None

    def __bool__(self) -> bool:
        return self.unavailable is None


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "ytx"


@lru_cache(maxsize=1)
def _vision_binary() -> Path | None:
    """Compile the Vision helper once and cache it; None if unsupported."""
    if sys.platform != "darwin" or not _SWIFT_SOURCE.is_file():
        return None
    if shutil.which("swiftc") is None:
        return None
    target = _cache_dir() / "vision_ocr"
    if target.is_file() and target.stat().st_mtime >= _SWIFT_SOURCE.stat().st_mtime:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["swiftc", "-O", str(_SWIFT_SOURCE), "-o", str(target)],
            capture_output=True,
            timeout=_COMPILE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return target if result.returncode == 0 and target.is_file() else None


@lru_cache(maxsize=1)
def available_engine() -> OcrEngine:
    """Pick the best OCR engine this machine can run."""
    if _vision_binary() is not None:
        return OcrEngine("apple-vision")
    if shutil.which("tesseract") is not None:
        return OcrEngine("tesseract")
    return OcrEngine("none", unavailable="no OCR engine (install tesseract, or use macOS)")


def _plausible(token: str) -> bool:
    """Is this a real word or number, rather than OCR debris?"""
    if _NUMERIC_TOKEN.match(token):
        return True
    if not _WORD_TOKEN.match(token):
        return False
    if token.isupper() and len(token) <= 5:
        return True  # an acronym: OOS, CAGR, AAPL
    return bool(_VOWEL.search(token))


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _vision_lines(path: Path) -> list[str]:
    binary = _vision_binary()
    if binary is None:
        return []
    output = _run([str(binary), str(path)])
    if not output:
        return []

    boxes: list[tuple[float, float, float, str]] = []
    for row in output.splitlines():
        fields = row.split("\t")
        if len(fields) != 6:
            continue
        try:
            x, y, height = float(fields[0]), float(fields[1]), float(fields[3])
        except ValueError:
            continue
        text = fields[5].strip()
        if text:
            boxes.append((x, y, height, text))

    # Top-to-bottom (Vision's origin is bottom-left), then left-to-right.
    boxes.sort(key=lambda box: (-box[1], box[0]))
    lines: list[str] = []
    current: list[tuple[float, str]] = []
    baseline: float | None = None
    for x, y, height, text in boxes:
        tolerance = max(height * _BASELINE_TOLERANCE, _MIN_BASELINE_TOLERANCE)
        if baseline is None or abs(y - baseline) <= tolerance:
            current.append((x, text))
            baseline = y if baseline is None else baseline
        else:
            lines.append(" ".join(part for _x, part in sorted(current)))
            current, baseline = [(x, text)], y
    if current:
        lines.append(" ".join(part for _x, part in sorted(current)))
    return lines


def _tesseract_lines(path: Path) -> list[str]:
    output = _run(["tesseract", str(path), "-", "--psm", _TESSERACT_PSM, "tsv"])
    if not output:
        return []
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for row in output.splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) < 12:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        token = fields[11].strip()
        # Tesseract emits far more debris than Vision, so its output is
        # additionally shape-filtered; Vision's is clean enough without it.
        if confidence < _MIN_CONFIDENCE or not token or not _plausible(token):
            continue
        grouped.setdefault((fields[2], fields[3], fields[4]), []).append(token)
    return [" ".join(tokens) for tokens in grouped.values()]


def ocr_lines(path: Path) -> list[str]:
    """Read a frame with the best available engine, as visual lines."""
    engine = available_engine()
    if not engine:
        return []
    lines = _vision_lines(path) if engine.name == "apple-vision" else _tesseract_lines(path)
    collapsed = (" ".join(line.split()) for line in lines)
    return [line for line in collapsed if len(line) > 4]
