# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- Refactored `src/transcribe.py` from single openai-whisper engine to multi-engine architecture with `--engine` CLI flag
- Replaced `print()` calls with structured `logging` in transcription module
- Lazy-loaded ML libraries (torch, whisper, faster_whisper, pywhispercpp) to avoid unnecessary imports

### Removed

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
