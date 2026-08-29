# ytx — YouTube Transcript Extractor

`ytx` turns a YouTube video into a document an LLM can consume: the raw
transcript, minus sponsor reads, with jargon repaired and on-screen chart text
folded in at the right timestamp. Everything is local and self-hosted.

```bash
uv sync --all-packages

ytx doc "<url>"            # AI-ready Markdown to stdout
ytx doc "<url>" --index    # write the doc, print a compact map of it instead
ytx doc "<url>" --frames   # also OCR charts and demos shown on screen
ytx get "<url>" --format srt   # raw export: json | srt | vtt | txt | md
ytx langs "<url>"              # available caption languages
ytx packs                     # bundled vocabulary packs
ytx health                    # circuit-breaker state per backend

uv run uvicorn ytx_api.main:app --port 8000   # optional REST API
```

## What `ytx doc` adds over `ytx get`

`ytx get` exports the transcript faithfully. `ytx doc` applies three cleanup
passes on top, each independently.

| Pass | Effect | Off with |
| --- | --- | --- |
| Sponsor removal | Drops ad reads and channel plugs as whole blocks | `--keep-sponsors`, `--no-sponsorblock` |
| Vocabulary repair | `sharp ratio` → `Sharpe ratio`, `engine x` → `nginx` | `--no-fix-terms`, `--domain none` |
| On-screen text | OCRs charts and dashboards into the transcript | on only with `--frames` |

`--no-clean` turns off all three. Everything removed or changed is reported in
a document banner and on stderr, with timestamps and the source that flagged it.

## `--index`

A long video is thousands of tokens to read in full. `--index` writes the
document to disk and prints only a map of it — file path, length, segment
count, token estimate, cleanup notes and a timestamped outline — so an agent
can `grep` the relevant part instead of reading everything.

```bash
ytx doc "<url>" --index
grep -n "drawdown" <video_id>.md
```

## `--frames` and on-screen OCR

Charts and demos carry substance the transcript never voices. `--frames`
downloads the video once, samples frames at scene changes (detected from
keyframes in a couple of seconds), OCRs the text, and inlines it under the
timestamps it was shown at. Sampling follows the video's own cuts rather than a
fixed interval, and is capped by `--max-frames` (default 40) so a static
screencast is not re-read forty times.

OCR uses Apple Vision where available, tesseract otherwise — both need
`ffmpeg` (and swiftc on macOS, or `tesseract` elsewhere). On a 720p frame the
difference is large enough to matter:

| engine | native | 2x upscaled | time |
| --- | --- | --- | --- |
| tesseract `--psm 11` | 56% | 75% | 3.1s |
| Apple Vision | 95% | 97% | 0.7s |

OCR output is noisy — treat it as approximate, not quotable. Frames and the
downloaded video live in a temporary directory and are deleted on return.

## Vocabulary packs

Jargon repairs live in TOML packs under
[`cleanup/packs/`](packages/ytx_core/src/ytx_core/cleanup/packs), one per
domain. Domain packs are selected from the video's own text; the `general` pack
always applies. Adding a domain needs no code change — drop in a file with
`name`, `detect` keywords and `rules`. Order matters: rules compile into one
regex, so the first alternative matching at a position wins; list `back testing`
before `back test`.

```bash
ytx doc "<url>" --domain trading,tech    # choose explicitly
ytx doc "<url>" --lexicon ./mine.toml    # your own pack, applied before bundled
```

## Sponsor removal

Two sources merge over the timeline:

- **SponsorBlock** — crowd-verified ranges, queried through its hash-prefix
  endpoint so only four hex characters of the video id's SHA-256 leave the
  machine.
- **Local keyword detection** — for videos SponsorBlock has never seen.

Detection is deliberately conservative: a local block needs two distinct
signal categories (or one strong one — a discount code or named sponsor),
bounds expand only over segments naming the advertiser, and removal is capped
both per block and per transcript. An ad is more likely to survive than real
content be removed.

## REST API

`uv run uvicorn ytx_api.main:app --port 8000` exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/transcripts/{video_id}` | One transcript. `format=json\|srt\|vtt\|txt\|md`, `languages=en,de`, `translate=en`, `refresh=true`, `download=true`. |
| `GET /api/v1/videos/{video_id}` | Available caption languages. |
| `POST /api/v1/transcripts` | Batch — body `{"urls": [...]}` → `{job_id}`. Populates the cache. |
| `GET /api/v1/jobs/{job_id}` | Batch job status + per-url results. |
| `GET /health` | Status and per-backend circuit-breaker state. |

The transcript endpoint can also run the full `doc` pipeline (same cleanup as
`ytx doc`) and return Markdown:

- `clean=true` — remove sponsor blocks and repair vocabulary; returns the
  composed Markdown document (frontmatter, chapters, cleanup banner).
- `frames=true` — additionally OCR on-screen text (needs `ffmpeg` + an OCR
  engine; degrades gracefully when absent). Downloads the video, so it's slow.
- `keep_sponsors=true`, `sponsorblock=false`, `fix_terms=false` — turn off
  individual cleanup passes.

```bash
curl "http://localhost:8000/api/v1/transcripts/VIDEO_ID?clean=true"
```

## Architecture

```
URL ─► resolver ─► cache? ─► cascade ─► captions_api  (youtube-transcript-api)
                                     └► audio + ASR   (YTX_ENABLE_ASR=1)
                    ─► exporters ─► json/srt/vtt/txt/md
                    └► doc ─► cleanup + on-screen OCR ─► md
```

The cascade tries backends in order, and each backend has its own circuit
breaker (5 consecutive failures trips it; a definitive "no captions" doesn't).
Cleanup runs after the cache, so the cache holds raw transcripts and disabling
cleanup never costs a re-fetch.

| Variable | Default | Purpose |
| --- | --- | --- |
| `YTX_DB_PATH` | `./ytx_cache.sqlite3` | Cache database |
| `YTX_DISABLE_CACHE` | unset | `1`/`true` to disable |
| `YTX_ENABLE_ASR` | unset | `1`/`true` for the Whisper backend |

```bash
uv run pytest -q      # 351 tests, offline
uv run ruff check .
```

## Legal note

Automated access is contrary to YouTube's ToS (contractual). Intended for
personal, research and accessibility use from your own connection; don't
redistribute transcripts commercially.