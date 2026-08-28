# Changelog

All notable changes to `/watch` are documented here.

## [Unreleased] — CMS fork (Colormestadic/claude-video)

Fork of `bradautomates/claude-video`. Upstream is unchanged except where listed.

### Added
- **`local` Whisper backend.** Shells out to the `whisper` CLI: no API key, no network, no cost, and
  the audio never leaves the machine. Model from `WATCH_LOCAL_MODEL` (default `small.en`); binary
  from `WATCH_LOCAL_WHISPER_BIN` when it is not on `PATH`. Force it with `--whisper local`.
  Rationale: Instagram, TikTok and most short-form sources ship no native captions, so they always
  fall through to Whisper. API-only made the platforms we use most the exact case that required a
  paid key and shipped audio to a third party.
- Backend resolution helper `resolve_backend()`, replacing direct `load_api_key()` calls at the
  `watch.py` call site. Order is Groq, then OpenAI, then local: an existing key keeps upstream
  behaviour (large-v3 beats a small local model), and local is the guaranteed keyless fallback.
- `setup.py` counts local whisper as satisfying the transcription gate, so a machine that can
  already transcribe for free is never nagged to buy an API key. New `local_whisper` field in
  `--json`; `has_api_key` stays truthful and is not set by the local backend.
- **GATE 0 in `SKILL.md`.** House rule: short-form social video (Reel, TikTok, Short, competitor
  clip) routes to `/cms-content-cloner` for a DNA teardown instead. `/watch` is an ingestion layer
  and may be called *by* that skill, never instead of it. It still owns screen recordings, bug
  repros, Looms, meetings, tutorials, podcasts and long-form playback outright.
- Tests: local backend resolution and precedence, real-CLI JSON parsing, failure surfacing, model
  configurability, and the setup gate. Existing keyless setup tests now pin
  `WATCH_LOCAL_WHISPER_BIN` off so they stay meaningful on a machine that has whisper installed.

### Fixed (post-review, 2026-08-28)
- **`--start/--end` now trims the audio before transcription** instead of transcribing the whole file
  and filtering the result. Focusing a section of a long captionless file previously saved nothing;
  on the CPU-bound local backend that was the difference between seconds and many minutes. Segments
  are shifted back into absolute source time, so callers see no change.
- **Default local model is the multilingual `small`, not `small.en`.** Whisper's `.en` models force
  English rather than detecting it, so a keyless machine silently returned garbage for any
  non-English source. English-only is now opt-in via `WATCH_LOCAL_MODEL=small.en`.
- **`setup.py` run directly no longer reports a working local-only install as a failed setup.**
  `cmd_install` checked `_have_api_key()` alone, so it printed "one step left: add a Whisper API key"
  and exited 3 on a machine that could already transcribe. It now uses the same
  `has_key or has_local` rule as `_status()`.
- **GATE 0 no longer routes to a skill that may not exist.** It named `/cms-content-cloner`
  unconditionally, which is not bundled here and is named differently on Codex, so on other installs
  the gate stopped `/watch` and left no path at all. It now looks for a teardown skill on the current
  host and, when none exists, proceeds while labelling the output ingestion rather than analysis.

### Changed
- Local audio is never chunked. Chunking exists only to stay under the APIs' 25 MB upload cap; the
  local backend uploads nothing and windows long audio itself, so splitting would add seams for no
  benefit.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable transcription entirely (frames only).
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.
