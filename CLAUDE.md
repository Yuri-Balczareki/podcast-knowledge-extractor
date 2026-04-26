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
python src/scraper.py                          # Phase 1: sync feed + download all pending
python src/scraper.py --sync-only              # sync feed to CSV without downloading
python src/scraper.py --limit 5                # download up to 5 pending episodes
python src/transcribe.py <audio.mp3> --engine faster-whisper --model base --language pt
python src/batch_transcribe.py                     # batch transcribe all pending episodes
python src/batch_transcribe.py --dry-run           # preview pending episodes
python src/batch_transcribe.py --limit 5 --workers 2  # parallel batch with limit
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

**Data flow:** RSS feed → `scraper.py` → MP3 in `data/audio/` → `transcribe.py` → JSON/TXT in `data/transcripts/` → `diarize.py` → `*.diarized.json/txt` with speaker labels.

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
- **Changelog**: update `CHANGELOG.md` at every commit via the `/create-commit` skill. Each commit's changes go under a dated section (`## [Unreleased] - YYYY-MM-DD`) with three subsections: `### Added`, `### Changed`, `### Removed`. Multiple commits on the same day append to the existing dated section. Entries are derived from the commit message summary. Skip trivial changes like comment edits or minor formatting.
- **README sync**: after every implementation that adds, changes, or removes user-facing functionality, review `README.md` and update it. This includes new scripts, new CLI flags, changed usage patterns, new dependencies, modified project structure, or removed features. The README must always reflect the current state of the project.
- **Test coverage**: every source file under `src/` with complex functions must have a corresponding unit test file under `tests/unit/`. When adding or modifying complex logic, ensure tests exist and cover both happy path and edge cases.

## Behavioral Guidelines

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
