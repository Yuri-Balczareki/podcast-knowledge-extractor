# Requirements: Batch Transcription Pipeline

> What must be true for this plan to succeed.

**Date:** 2026-04-25

---

## Dependencies

- **pywhispercpp >= 1.4** — GGML-based Whisper with Metal GPU support (already in requirements.txt)
- **ffprobe** — Required for audio duration detection (used by transcribe.py, must be on PATH)
- **concurrent.futures** — stdlib, no additional install needed

## Prerequisites

- [ ] Audio files downloaded via `python src/scraper.py` (at least some episodes with `status=downloaded`)
- [ ] whisper.cpp large-v3 model available (auto-downloads on first run via pywhispercpp, ~3GB)
- [ ] Sufficient disk space for transcript outputs (~10KB per episode in data/transcripts/)

## Acceptance Criteria

- [ ] CSV gains `transcription_status` and `transcript_path` columns without breaking existing scraper workflow
- [ ] `--dry-run` lists pending episodes without modifying any files
- [ ] Sequential mode (`--workers 1`) transcribes episodes and updates CSV after each one
- [ ] Parallel mode (`--workers 2`) runs multiple whisper.cpp instances concurrently
- [ ] Re-running skips already-transcribed episodes (resume support)
- [ ] Existing transcripts (e.g., NerdCast 1025) are auto-detected and marked in CSV
- [ ] Failed transcriptions are marked as `failed` and retried on next run

## Constraints

- Default to sequential (1 worker) for safety; parallel is opt-in via `--workers`
- CSV must be saved after each episode completion for crash recovery
- Existing `transcribe.py` CLI must remain unchanged (backward compatible)
- Model is loaded once per worker process, not per file
