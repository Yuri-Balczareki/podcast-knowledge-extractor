# Validation: Batch Transcription Pipeline

> How to verify this plan succeeded.

**Date:** 2026-04-25

---

## Automated Checks

- [ ] `pytest tests/unit/test_batch_transcribe.py -v` — all batch logic tests pass
- [ ] `ruff check src/ tests/` — no lint errors
- [ ] `python src/transcribe.py data/audio/<any>.mp3 --engine whisper.cpp --model base` — existing CLI still works

## Manual Verification

1. **CSV schema update** — Run `python src/scraper.py --sync-only`, verify CSV has 11 columns with `transcription_status` and `transcript_path`
2. **Auto-detect** — Run `python src/batch_transcribe.py --dry-run`, verify NerdCast 1025 does NOT appear (already transcribed, auto-detected)
3. **Dry run** — Verify `--dry-run` prints pending list without modifying CSV or creating files
4. **Single transcription** — Run `python src/batch_transcribe.py --limit 1`, verify:
   - CSV row updated with `transcription_status=transcribed` and `transcript_path` pointing to JSON
   - JSON and TXT files created in `data/transcripts/`
5. **Resume** — Re-run `--dry-run`, verify the just-transcribed episode no longer appears
6. **Parallel mode** — Run `python src/batch_transcribe.py --limit 2 --workers 2`, verify both complete

## Expected Outputs

- `data/transcripts/{episode_stem}.json` — segment-level transcription for each processed episode
- `data/transcripts/{episode_stem}.txt` — full text transcription
- `data/jovem-nerd-episodes.csv` — updated with transcription status and paths for processed episodes

## Rollback

- Delete `src/batch_transcribe.py` and `tests/unit/test_batch_transcribe.py`
- Revert `CSV_FIELDS` change in `src/scraper.py` (remove 2 fields)
- Revert model reuse functions in `src/transcribe.py`
- CSV data: the new columns are harmless but can be removed by re-running scraper after reverting
