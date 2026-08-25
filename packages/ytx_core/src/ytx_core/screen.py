"""On-screen text extraction: OCR chart and dashboard frames a talk shows but never says.

Sampling follows scene changes (from keyframes), is capped by ``max_frames``,
and is failure-tolerant so a missing tool or bad frame degrades the result
instead of raising. Frames are transient; only the OCR text survives inline.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ytx_core.ocr import available_engine, ocr_lines

__all__ = ["ScreenCapture", "ScreenTextResult", "extract_screen_text", "tools_available"]

_FORMAT = "bv[height<=720][protocol^=https]"
# Scene-change score above which two keyframes show different content. The
# 0.3-0.4 usually quoted is for film cuts; screen content changes far more
# subtly, and at 0.3 a slide deck registers nothing at all.
_SCENE_THRESHOLD = 0.045
# Comparing keyframes rather than every frame makes detection ~10x faster and
# still resolves changes to within a GOP, which is ample for slides and UIs.
_SCENE_TIMEOUT = 180.0
# Fallback spacing when scene detection is unavailable (streaming path).
_FALLBACK_INTERVAL = 120.0
# Past this, a full download is too much transient disk to be polite; fall back
# to seeking the stream directly and accept the slower run.
_MAX_DOWNLOAD_SECONDS = 4 * 3600
_DOWNLOAD_CONNECTIONS = "8"
_DOWNLOAD_TIMEOUT = 600.0
# A frame carrying at least this many OCR words is showing a screen, not a face.
_SCREEN_WORD_THRESHOLD = 25
_DEFAULT_INTERVAL = 120.0
_DEFAULT_MAX_FRAMES = 40
# Upscaling at grab time costs nothing extra and measurably lifts recall on
# small UI text.
_UPSCALE = 2
_FFMPEG_TIMEOUT = 30.0
_TESSERACT_TIMEOUT = 30.0
_YTDLP_TIMEOUT = 60.0
# Lines present in more than this share of captures are static chrome: a
# watermark, a sponsor banner, a window title, a sidebar. Kept below half
# because a screen-share often covers only part of a video, so its chrome
# recurs across the captures that exist rather than across the whole run.
_BOILERPLATE_SHARE = 0.35
# A line at least this proportion digits is data, and never boilerplate.
_NUMERIC_LINE_SHARE = 0.3
# Edit-similarity at which two OCR readings are the same underlying line.
# Short strings score high by coincidence, so only long signatures are
# compared fuzzily; shorter ones must match exactly.
_SIGNATURE_SIMILARITY = 0.90
_MIN_FUZZY_LENGTH = 20
# Two consecutive captures sharing this much vocabulary are the same screen.
_DUPLICATE_OVERLAP = 0.85
_MAX_CAPTURE_CHARS = 420
# Below this a capture is a stray caption or a logo, not a screen worth noting.
_MIN_CAPTURE_CHARS = 60

# Tesseract reports per-word confidence; below this it is usually noise, and
# much above it the small text on a screen recording starts being discarded.
_MIN_CONFIDENCE = 45.0
_WORD_RE = re.compile(r"[A-Za-z0-9][\w.$%/-]*")
# A number, price, percentage or date — the part of a chart worth recovering.
_NUMERIC_TOKEN = re.compile(r"^[$€£]?\d[\d.,:%/-]*[%a-zA-Z]?$")
_WORD_TOKEN = re.compile(r"^[A-Za-z][A-Za-z'./-]{2,}$")
_VOWEL = re.compile(r"[aeiouAEIOU]")


@dataclass(frozen=True)
class ScreenCapture:
    time: float
    text: str


@dataclass(frozen=True)
class ScreenTextResult:
    captures: list[ScreenCapture]
    #: Empty when extraction ran cleanly; otherwise why it produced less.
    status: str = ""

    def __bool__(self) -> bool:
        return bool(self.captures)


def tools_available() -> str | None:
    """Return the name of the first missing external tool, or None."""
    if shutil.which("ffmpeg") is None:
        return "ffmpeg"
    engine = available_engine()
    return None if engine else "tesseract"


def _stream_url(video_id: str) -> str | None:
    """A directly seekable URL for the video — the slow path."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", "-f", _FORMAT, "-g",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=_YTDLP_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _download(video_id: str, target: Path) -> Path | None:
    """Fetch the video once with parallel range requests."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", "-f", _FORMAT,
                "-N", _DOWNLOAD_CONNECTIONS,
                "--no-part", "--no-playlist", "--quiet", "--no-warnings",
                "-o", str(target),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            timeout=_DOWNLOAD_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        return None
    return target


def _scene_cuts(path: str, workdir: Path) -> list[float] | None:
    """Timestamps where the picture changes, from keyframes only.

    ``-skip_frame nokey`` compares consecutive keyframes instead of decoding
    every frame, which is what keeps this to a couple of seconds on a
    feature-length video. Returns None when detection could not run.
    """
    report = workdir / "scenes.txt"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-skip_frame", "nokey", "-i", path,
                "-vf",
                f"select='gt(scene,{_SCENE_THRESHOLD})',metadata=print:file={report}",
                "-fps_mode", "passthrough", "-f", "null", "-",
            ],
            capture_output=True,
            timeout=_SCENE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not report.is_file():
        return None
    cuts = [
        float(match.group(1))
        for match in re.finditer(
            r"pts_time:([\d.]+)", report.read_text(encoding="utf-8", errors="replace")
        )
    ]
    return sorted(set(cuts))


def _segments(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    bounds = [0.0, *(cut for cut in cuts if 0.0 < cut < duration), duration]
    return [
        (start, end) for start, end in zip(bounds, bounds[1:], strict=False) if end > start
    ]


def _moments(
    segments: list[tuple[float, float]], interval: float, max_frames: int
) -> list[float]:
    """Where to sample: the middle of each segment, plus more inside long ones.

    Sampling the midpoint rather than the cut matters — a frame taken at a cut
    lands mid-transition, on a fade or a half-drawn build.

    When there are more candidates than budget, the longest segments win rather
    than an even spread across the video. A screen someone is talking over
    holds still for tens of seconds; conversation is cut every few. Measured on
    an 85-minute interview with a software demo in its second half, 77% of
    segments longer than 15s fell in the demo. An even spread would instead
    spend a third of the budget on faces.
    """
    candidates: list[tuple[float, float]] = []  # (segment duration, moment)
    for start, end in segments:
        span = end - start
        if span <= interval:
            candidates.append((span, start + span / 2))
            continue
        at = start + interval / 2
        while at < end:
            candidates.append((span, at))
            at += interval
    if len(candidates) <= max_frames:
        return sorted(moment for _span, moment in candidates)
    candidates.sort(key=lambda item: -item[0])
    return sorted(moment for _span, moment in candidates[:max_frames])


def _source(video_id: str, duration: float, workdir: Path) -> tuple[str | None, bool]:
    """Resolve what ffmpeg should read: a local copy if worth downloading."""
    if duration <= _MAX_DOWNLOAD_SECONDS:
        local = _download(video_id, workdir / "video.mp4")
        if local is not None:
            return str(local), True
    return _stream_url(video_id), False


def _grab(url: str, seconds: float, target: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error",
                "-ss", f"{seconds:.2f}", "-i", url,
                "-frames:v", "1", "-q:v", "2",
                "-vf", f"scale=iw*{_UPSCALE}:ih*{_UPSCALE}:flags=lanczos",
                str(target), "-y",
            ],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and target.is_file() and target.stat().st_size > 0


def _word_count(lines: list[str]) -> int:
    return sum(len(_WORD_RE.findall(line)) for line in lines)


def _signature(line: str) -> str:
    """Letters only, lowercased — the shape of a line without its numbers."""
    return re.sub(r"[^a-z]+", "", line.lower())


def _is_numeric_row(line: str) -> bool:
    """Digit-heavy lines are data, not chrome, even when their words repeat.

    "Builder Progress Full settings Results" is chrome on every frame. A table
    row sharing the word "Strategy" but carrying different numbers is not, so
    matching must never collapse the two.
    """
    if not line:
        return False
    digits = sum(character.isdigit() for character in line)
    return digits >= len(line) * _NUMERIC_LINE_SHARE


def _canonicalise(signatures: Iterable[str]) -> dict[str, str]:
    """Map near-identical signatures onto one representative.

    OCR reads the same title bar slightly differently every frame
    ("ChartFonatics", "ChartFanatics", "ChartFonotics"), so exact keys never
    match. Comparing by edit-similarity collapses them; the candidate list
    stays short because chrome repeats and content does not.
    """
    canon: dict[str, str] = {}
    known: list[str] = []
    for signature in signatures:
        if signature in canon:
            continue
        match = None
        for candidate in known if len(signature) >= _MIN_FUZZY_LENGTH else ():
            if abs(len(candidate) - len(signature)) > max(len(signature) * 0.25, 3):
                continue
            if SequenceMatcher(None, candidate, signature).ratio() >= _SIGNATURE_SIMILARITY:
                match = candidate
                break
        canon[signature] = match or signature
        if match is None:
            known.append(signature)
    return canon


def _strip_boilerplate(captures: list[tuple[float, list[str]]]) -> list[tuple[float, list[str]]]:
    """Thin out lines that recur across captures — watermarks, banners, chrome.

    A recurring line is kept where it first appears and dropped afterwards, so
    a persistent title bar is stated once rather than eighteen times, while a
    table header that happens to stay on screen is not lost entirely.
    """
    if len(captures) < 3:
        return captures

    canon = _canonicalise(
        _signature(line) for _time, lines in captures for line in lines if _signature(line)
    )
    counts = Counter(
        key
        for _time, lines in captures
        for key in {canon[_signature(line)] for line in lines if _signature(line)}
    )
    ceiling = max(len(captures) * _BOILERPLATE_SHARE, 2)

    seen: set[str] = set()
    thinned: list[tuple[float, list[str]]] = []
    for time, lines in captures:
        kept: list[str] = []
        for line in lines:
            signature = _signature(line)
            if not signature:
                continue
            key = canon[signature]
            if counts[key] > ceiling and key in seen and not _is_numeric_row(line):
                continue
            kept.append(line)
            seen.add(key)
        thinned.append((time, kept))
    return thinned


def _drop_repeats(captures: list[tuple[float, list[str]]]) -> list[tuple[float, list[str]]]:
    """Collapse consecutive captures of an unchanged screen."""
    kept: list[tuple[float, list[str]]] = []
    previous: set[str] = set()
    for time, lines in captures:
        words = {word.lower() for line in lines for word in _WORD_RE.findall(line)}
        if not words:
            continue
        if previous:
            overlap = len(words & previous) / max(len(words | previous), 1)
            if overlap >= _DUPLICATE_OVERLAP:
                continue
        kept.append((time, lines))
        previous = words
    return kept


def extract_screen_text(
    video_id: str,
    duration: float,
    *,
    interval: float = _DEFAULT_INTERVAL,
    max_frames: int = _DEFAULT_MAX_FRAMES,
    keep_frames: Path | None = None,
) -> ScreenTextResult:
    """OCR the on-screen text of a video.

    Sampling follows the video's own scene changes, so a static screencast is
    not resampled forty times and a screen that appears only briefly is not
    missed between fixed intervals. Frames live in a temporary directory
    unless ``keep_frames`` is given, and are deleted either way on return.
    """
    if shutil.which("ffmpeg") is None:
        return ScreenTextResult([], "ffmpeg is not installed")
    engine = available_engine()
    if not engine:
        return ScreenTextResult([], engine.unavailable or "no OCR engine")
    if duration <= 0:
        return ScreenTextResult([], "unknown duration")

    with tempfile.TemporaryDirectory(prefix="ytx-frames-") as scratch:
        workdir = Path(scratch)
        url, local = _source(video_id, duration, workdir)
        if url is None:
            return ScreenTextResult([], "could not resolve a video stream")

        # Scene detection needs a local file to be cheap. Streaming falls back
        # to one segment spanning the video, which samples at a fixed interval.
        cuts = _scene_cuts(url, workdir) if local else None
        segments = _segments(cuts, duration) if cuts is not None else [(0.0, duration)]
        moments = _moments(segments, interval, max_frames)
        if not moments:
            return ScreenTextResult([], "nothing to sample")

        raw: list[tuple[float, list[str]]] = []
        read = 0
        for index, at in enumerate(moments):
            frame = workdir / f"frame-{index}.jpg"
            if not _grab(url, at, frame):
                continue
            read += 1
            lines = ocr_lines(frame)
            if keep_frames is not None:
                keep_frames.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame, keep_frames / f"{video_id}-{int(at)}.jpg")
            frame.unlink(missing_ok=True)
            # Faces and b-roll read as a handful of stray words; a screen does not.
            if _word_count(lines) >= _SCREEN_WORD_THRESHOLD:
                raw.append((at, lines))
        if read == 0:
            return ScreenTextResult([], "no frames could be read")

    truncated = len(moments) >= max_frames
    cleaned = _drop_repeats(
        [(time, lines) for time, lines in _strip_boilerplate(raw) if lines]
    )
    captures = [
        capture
        for capture in (
            ScreenCapture(time=time, text=" · ".join(lines)[:_MAX_CAPTURE_CHARS])
            for time, lines in cleaned
            if lines
        )
        if len(capture.text) >= _MIN_CAPTURE_CHARS
    ]
    status = f"stopped at the {max_frames}-frame cap" if truncated else ""
    if not captures:
        status = status or "no on-screen text found"
    return ScreenTextResult(captures, status)
