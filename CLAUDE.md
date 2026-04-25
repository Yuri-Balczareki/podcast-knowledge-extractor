# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Podcast knowledge extraction pipeline: download episodes via RSS, transcribe audio (3 Whisper engines), identify speakers (pyannote.audio), and build a searchable knowledge base with RAG. Phases 1–3 are implemented; Phases 4–6 (vector indexing, RAG, Streamlit UI) are planned.

Content is Brazilian Portuguese (Nerdcast podcast). The transcription initial prompt preserves proper nouns (Alottoni, Azaghal, NerdCast) and English loanwords.

## Commands

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline stages
python src/downloader.py                          # Phase 1: interactive episode download
python src/transcribe.py <audio.mp3> --engine faster-whisper --model base --language pt
python src/diarize.py <audio.mp3>                 # requires HF_TOKEN in .env

# Tests
pytest tests/                                     # all tests
pytest tests/unit/test_diarize.py -v              # single file
pytest tests/unit/test_diarize.py::TestMerge -v   # single class

# Linting
ruff check src/ tests/
ruff format src/ tests/

# Benchmarking
python scripts/compare_whisper.py <audio.mp3> --model base --engines openai-whisper faster-whisper whisper.cpp --save-json
```

## Architecture

**Data flow:** RSS feed → `downloader.py` → MP3 in `data/audio/` → `transcribe.py` → JSON/TXT in `data/transcripts/` → `diarize.py` → `*.diarized.json/txt` with speaker labels.

Each module is a standalone CLI script (`argparse` + `if __name__ == "__main__": main()`).

**Transcription engines** — three engines behind a unified `transcribe()` interface returning `{"text", "segments", "language"}`:
- `openai-whisper` — reference implementation
- `faster-whisper` — CTranslate2, 4x faster (default)
- `whisper.cpp` — GGML, native Metal GPU via `pywhispercpp`

**Device auto-detection** pattern used in both `transcribe.py` and `diarize.py`: MPS → CUDA → CPU.

**Diarization merge** (`diarize.merge()`) — O(N×M) temporal overlap algorithm that assigns each transcript segment the speaker with maximum overlap duration. No overlap → `UNKNOWN`.

## Environment

- Python 3.12+, optimized for Apple Silicon (MPS/Metal)
- `HF_TOKEN` in `.env` — required for pyannote.audio speaker diarization models
- whisper.cpp model binary lives in `models/ggml-base.bin`

## Key Conventions

- Paths resolved relative to project root via `Path(__file__).resolve().parent.parent`
- Output directories: `data/audio/`, `data/transcripts/`, `data/chroma/` (all gitignored)
- Benchmarking results go in `data/comparisons/`
- Design docs live in `docs/features/`
- **Changelog**: every significant code change (new features, bug fixes, refactors, dependency changes) must be summarized under `## [Unreleased]` in `CHANGELOG.md` using [Keep a Changelog](https://keepachangelog.com) categories (`Added`, `Changed`, `Fixed`, `Removed`). Skip trivial changes like comment edits or minor formatting.
