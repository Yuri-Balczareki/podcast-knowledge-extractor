# Validation: Batch Diarization Integration

## Automated Checks

1. **Lint** — `ruff check src/ tests/` passes with no errors
2. **Tests** — `pytest tests/unit/test_batch_transcribe.py -v` passes (all existing + new tests)
3. **Full suite** — `pytest tests/` passes with no regressions

## Manual Validation

4. **Dry-run with diarize** — `python src/batch_transcribe.py --diarize --dry-run` shows pending episodes without errors
5. **Skip-transcription dry-run** — `python src/batch_transcribe.py --skip-transcription --diarize --dry-run` lists transcribed-but-not-diarized episodes
6. **Missing HF_TOKEN** — unset `HF_TOKEN` and run `python src/batch_transcribe.py --diarize --dry-run`; should warn and proceed without crashing
7. **Skip-transcription without diarize** — `python src/batch_transcribe.py --skip-transcription` warns "requires --diarize" and exits gracefully

## Acceptance Criteria

- [ ] `--diarize` flag produces `.diarized.json` files with `"speaker"` field on each segment
- [ ] CSV updates with `diarization_status` and `diarized_transcript_path` after each episode
- [ ] Existing transcription-only workflow (`python src/batch_transcribe.py`) works identically to before
- [ ] Multi-worker mode + diarize works (transcription parallel, diarization sequential)
- [ ] All unit tests pass
