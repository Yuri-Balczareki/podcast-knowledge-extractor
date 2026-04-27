# Plan: Integrate Speaker Diarization into Batch Transcription Pipeline

  PYTHONWARNINGS=ignore python src/batch_transcribe.py --language pt --model large --duration-limit 5 --diarize --limit 1

> Add `--diarize` flag to `batch_transcribe.py` that runs pyannote speaker diarization after transcription, producing speaker-attributed transcript segments.

**Date:** 2026-04-26
**Status:** Planned

---

## Context

The project has a standalone `diarize.py` module (pyannote/speaker-diarization-3.1) that identifies speakers and merges labels into transcript segments via a max-overlap algorithm. However, diarization requires a manual separate step after transcription. The goal is to integrate it into `batch_transcribe.py` so a single command produces speaker-attributed transcripts (`{"start", "end", "text", "speaker"}`), tracked via the CSV like transcription already is.

## Approach

**Opt-in `--diarize` flag** with backwards-compatible defaults. When enabled:

1. Pyannote pipeline loads **once** before the episode loop (~10s load, ~110MB model).
2. After each successful transcription, diarization runs inline: load transcript JSON -> run diarization -> merge via max-overlap -> save `.diarized.json/.txt`.
3. Multi-worker mode: transcription workers run in parallel, then diarization runs sequentially in the main process (pyannote pipeline is not picklable).
4. A `--skip-transcription` flag allows diarizing already-transcribed episodes without re-transcribing.

**Full reuse of `diarize.py`** — all existing functions (`load_pipeline`, `diarize`, `merge`, `load_transcript`, `save_output`) are imported and called directly. Zero duplication.

**CSV tracking** — two new fields: `diarization_status` and `diarized_transcript_path`.

## Steps

### 1. Extend CSV schema (`src/scraper.py`)
- Add `"diarization_status"` and `"diarized_transcript_path"` to `CSV_FIELDS` after `transcript_path`
- Add `setdefault` calls in `sync_feed_to_csv` for backwards compatibility

### 2. Add imports and helpers to `src/batch_transcribe.py`

**New imports from `diarize.py`:**
```python
from src.diarize import (
    diarize as run_diarization,
    get_device,
    load_pipeline,
    load_transcript,
    merge,
    save_output as save_diarized_output,
)
```

**New functions:**
- `get_pending_diarization(episodes)` — filter: transcribed + not diarized/failed
- `detect_existing_diarized_transcripts(episodes)` — scan for `{stem}.diarized.json`
- `_diarize_episode(ep, audio_dir, pipeline, prefix)` — orchestrate diarize+merge+save for one episode

### 3. Modify `batch_transcribe()` function

**New parameters (backwards-compatible):**
```python
enable_diarization: bool = False,
skip_transcription: bool = False,
diarization_device: str | None = None,
```

**Logic additions:**
1. `setdefault` for `diarization_status` / `diarized_transcript_path` on each episode
2. Auto-detect existing diarized transcripts when `--diarize` is enabled
3. Load pyannote pipeline once; if `HF_TOKEN` missing, warn and disable diarization
4. `--skip-transcription` mode: find transcribed-but-not-diarized episodes, diarize them, return early
5. Single-worker: after each transcription success, call `_diarize_episode`
6. Multi-worker: after `as_completed` loop, sequential diarization pass
7. Summary log includes diarization counts

### 4. Add CLI arguments to `main()`
- `--diarize` (store_true)
- `--skip-transcription` (store_true)
- `--diarization-device` (cpu/cuda/mps, default: auto-detect)

### 5. Unit tests (`tests/unit/test_batch_transcribe.py`)

Extend `_make_episode` with diarization fields. Add test classes:
- `TestGetPendingDiarization` — filtering logic (6 cases)
- `TestDetectExistingDiarizedTranscripts` — auto-detection (4 cases)
- `TestDiarizationBackfillColumns` — setdefault behavior (2 cases)
- `TestBatchDiarizeSkipTranscription` — warns without `--diarize` (1 case)

### 6. Documentation updates
- `README.md` — new CLI examples
- `CLAUDE.md` — new commands section entries
- `CHANGELOG.md` — new dated entry

## Files to Change

| File | Action | Purpose |
|------|--------|---------|
| `src/scraper.py` | Edit | Add 2 fields to `CSV_FIELDS`, add `setdefault` calls |
| `src/batch_transcribe.py` | Edit | New imports, 3 helpers, modify `batch_transcribe()` + `main()` |
| `src/diarize.py` | No change | All functions reused as-is |
| `tests/unit/test_batch_transcribe.py` | Edit | Extend helper, add 4 test classes |
| `README.md` | Edit | New usage examples |
| `CLAUDE.md` | Edit | New commands |
| `CHANGELOG.md` | Edit | New entry |

## Functions Reused from `diarize.py` (no changes)

| Function | Location | Purpose |
|----------|----------|---------|
| `load_pipeline(hf_token, device)` | `diarize.py:28` | Load pyannote model once |
| `diarize(audio_path, pipeline)` | `diarize.py:45` | Get speaker segments |
| `load_transcript(json_path)` | `diarize.py:65` | Load transcript JSON |
| `merge(transcript, diarization)` | `diarize.py:72` | Max-overlap speaker assignment |
| `save_output(segments, audio_path)` | `diarize.py:87` | Write `.diarized.json` + `.diarized.txt` |
| `get_device()` | `diarize.py:18` | Auto-detect MPS/CUDA/CPU |

## Output Format

**Before (transcription only)** — `{stem}.json`:
```json
[{"start": 0.0, "end": 7.5, "text": "Fala pessoal..."}]
```

**After (with `--diarize`)** — `{stem}.diarized.json`:
```json
[{"start": 0.0, "end": 7.5, "text": "Fala pessoal...", "speaker": "SPEAKER_00"}]
```

## CLI Usage

```bash
# Transcribe + diarize in one pass
python src/batch_transcribe.py --diarize

# Diarize already-transcribed episodes (no re-transcription)
python src/batch_transcribe.py --skip-transcription --diarize

# Force CPU for diarization (e.g. MPS issues)
python src/batch_transcribe.py --diarize --diarization-device cpu

# Preview what would be processed
python src/batch_transcribe.py --diarize --dry-run

# Limit + diarize
python src/batch_transcribe.py --diarize --limit 5
```
