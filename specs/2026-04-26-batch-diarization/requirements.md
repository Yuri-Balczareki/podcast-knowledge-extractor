# Requirements: Batch Diarization Integration

## Functional Requirements

1. **Opt-in diarization** — `--diarize` flag enables speaker diarization after transcription. Default behavior unchanged.
2. **Single pipeline load** — pyannote model loads once and is reused across all episodes in the batch.
3. **CSV tracking** — new columns `diarization_status` ("not_diarized", "diarized", "failed") and `diarized_transcript_path` track diarization progress per episode.
4. **Auto-detection** — existing `.diarized.json` files are detected on startup (mirrors transcript auto-detection).
5. **Skip-transcription mode** — `--skip-transcription --diarize` allows diarizing already-transcribed episodes without re-transcribing.
6. **Graceful HF_TOKEN handling** — if `HF_TOKEN` is not set, warn and skip diarization (don't fail the batch).
7. **Dry-run support** — `--diarize --dry-run` shows what would be diarized without executing.

## Non-Functional Requirements

1. **Backwards compatibility** — existing transcription-only workflow is unaffected. Old CSVs get backfilled defaults.
2. **Memory** — Whisper large (~3GB) + pyannote (~110MB) coexist in single process on 32GB system.
3. **Multi-worker safety** — pyannote runs sequentially in the main process after parallel whisper.cpp workers finish (pipeline is not picklable).
4. **No code duplication** — all diarization logic reused from `diarize.py`.

## Constraints

- Python 3.12+, Apple Silicon (MPS/Metal)
- pyannote/speaker-diarization-3.1 model (requires HuggingFace token + model license acceptance)
- Speaker labels are anonymous: `SPEAKER_00`, `SPEAKER_01`, etc.
