# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-04-26

### Added

- Automated RSS scraper (`src/scraper.py`) with feed sync to CSV catalog and batch MP3 downloads via httpx
- Episode catalog (`data/jovem-nerd-episodes.csv`) with 1054 episodes from Jovem Nerd feed
- CLI modes for scraper: `--sync-only` (feed catalog only) and `--limit N` (controlled batch downloads)
- `specs/MISSION.md` for project mission statement
- Comprehensive README with project overview, architecture, usage, and roadmap
- Behavioral guidelines in CLAUDE.md (think before coding, simplicity first, surgical changes, goal-driven execution)
- README sync and test coverage conventions in CLAUDE.md
- Three-engine transcription support: faster-whisper, openai-whisper, and whisper.cpp behind a unified `transcribe()` interface
- Portuguese initial prompt for transcription to preserve proper nouns (Alottoni, Azaghal, NerdCast) and English loanwords
- Speaker diarization module (`src/diarize.py`) using pyannote.audio with temporal overlap merge algorithm
- Whisper engine benchmarking script (`scripts/compare_whisper.py`) with WER, memory, and timing metrics
- Unit tests for diarization merge algorithm (`tests/unit/test_diarize.py`)
- Speaker diarization design doc (`docs/features/speaker-diarization.md`)
- CLAUDE.md for Claude Code guidance
- `pywhispercpp>=1.4` dependency for native Metal GPU transcription on Apple Silicon
- `jiwer>=3.0` dependency for word error rate benchmarking
- `.gitignore` entries for `data/comparisons/`
- Batch transcription pipeline (`src/batch_transcribe.py`) with CSV-tracked progress, auto-detection of existing transcripts, resume support, and optional parallel workers via ProcessPoolExecutor
- Model reuse API in `src/transcribe.py` (`load_whisper_cpp_model`, `transcribe_with_model`) for multi-file efficiency
- `transcription_status` and `transcript_path` columns in episode CSV schema with backfill for existing rows
- Unit tests for batch transcription logic (`tests/unit/test_batch_transcribe.py`) and transcription routing (`tests/unit/test_transcribe.py`)
- Batch transcription spec documents (`specs/2026-04-25-batch-transcription/`)

### Changed

- Replaced `src/downloader.py` (interactive) with `src/scraper.py` (automated RSS sync + batch download)
- Reorganized project docs into `specs/` directory (`MISSION_ROADMAP.md` → `specs/ROADMAP.md`, `TECH_STACK.md` → `specs/TECH_STACK.md`)
- Expanded CLAUDE.md with updated commands, architecture references, and changelog/README conventions
- `.gitignore`: added `data/audio/*.part` for partial download files
- Refactored `src/transcribe.py` from single openai-whisper engine to multi-engine architecture with `--engine` CLI flag
- Replaced `print()` calls with structured `logging` in transcription module
- Lazy-loaded ML libraries (torch, whisper, faster_whisper, pywhispercpp) to avoid unnecessary imports
- Refactored whisper.cpp internals to support model reuse across multiple transcriptions with progress ETA logging
- Added worker-level logging in batch transcription parallel mode for model init observability

### Removed

- `src/downloader.py` (replaced by `src/scraper.py`)
- `whisperx>=3.1` dependency (replaced by standalone pyannote.audio for diarization)

## [0.1.0] - 2026-04-24

### Added

- Project scaffold with module structure (`src/`, `app/`, `data/`, `scripts/`, `tests/`)
- Podcast ingestion pipeline: RSS feed parsing and episode downloading (`src/downloader.py`)
- Audio transcription with faster-whisper, Apple Silicon optimized (`src/transcribe.py`)
- SQLite metadata storage for episodes (`src/db.py`)
- Streamlit web interface scaffold (`app/`)
- Project documentation: README, MISSION_ROADMAP, TECH_STACK
- Python dependencies and environment configuration

## [0.0.0] - 2026-04-23

### Added

- Initial repository setup with LICENSE and .gitignore
