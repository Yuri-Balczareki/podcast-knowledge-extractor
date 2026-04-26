# Plan: Batch Transcription Pipeline

> Orchestrate batch transcription of all downloaded podcast episodes with CSV tracking, auto-detection of existing transcripts, and optional parallel workers.

**Date:** 2026-04-25
**Status:** Planned

---

## Context

The project has ~1054 episodes in the CSV catalog, ~120+ downloaded audio files, but only 1 transcribed so far. Transcription is currently single-file CLI only (`python src/transcribe.py <file>`). We need a batch orchestration layer that processes all downloaded-but-not-transcribed episodes using whisper.cpp large model (fastest engine, Metal GPU on Apple Silicon), tracks progress in the CSV, and supports optional parallelization on a 32GB Mac.

## Approach

Three changes: extend the CSV schema with transcription tracking columns, refactor `transcribe.py` to support model reuse across multiple files, and create a new `batch_transcribe.py` CLI script.

**Parallelization**: `ProcessPoolExecutor` with per-worker model initialization (pywhispercpp Model is not picklable). Each worker loads its own ~3GB model. With 32GB unified RAM: 2 workers safe (~8GB), 3 feasible (~12GB). CPU threads split as `cpu_count // workers`.

## Steps

1. **Extend CSV schema** — Add `transcription_status` and `transcript_path` columns. Files: `src/scraper.py`
2. **Model reuse API** — Add `load_whisper_cpp_model()`, `transcribe_with_model()`, refactor `_transcribe_whisper_cpp()`. Files: `src/transcribe.py`
3. **Batch orchestrator** — New CLI with auto-detect, sequential/parallel modes, CSV tracking. Files: `src/batch_transcribe.py`
4. **Unit tests** — Test filtering, backfill, auto-detect, thread allocation. Files: `tests/unit/test_batch_transcribe.py`
5. **Documentation** — Update README.md, CLAUDE.md, CHANGELOG.md

## Files to Change

| File | Action | Purpose |
|------|--------|---------|
| `src/scraper.py` | Modify | Add `transcription_status`, `transcript_path` to CSV_FIELDS; backfill defaults |
| `src/transcribe.py` | Modify | Add `load_whisper_cpp_model()`, `transcribe_with_model()`, refactor internals |
| `src/batch_transcribe.py` | Create | Batch orchestration CLI with CSV tracking and parallel workers |
| `tests/unit/test_batch_transcribe.py` | Create | Unit tests for batch logic |
| `README.md` | Modify | Add batch transcription usage section |
| `CLAUDE.md` | Modify | Add batch transcription commands |
| `CHANGELOG.md` | Modify | Record changes |
