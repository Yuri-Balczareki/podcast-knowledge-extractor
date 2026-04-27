# Plan: Parallel Transcription + Diarization per Episode

> Run transcription and diarization concurrently for each episode, merging results when both complete. Reduces per-episode wall-clock time from `T_transcribe + T_diarize` to `max(T_transcribe, T_diarize)`.

**Date:** 2026-04-26
**Status:** Planned

---

## Context

The current pipeline processes each episode sequentially: transcribe → diarize → merge. But transcription and diarization are independent — both only read the audio file. On Apple Silicon, they naturally use different hardware: faster-whisper runs on CPU (int8), pyannote runs on MPS (GPU). Running them concurrently should yield near-free parallelism.

For a typical 1-hour episode:
- Transcription (faster-whisper large, CPU): ~30 min
- Diarization (pyannote 3.1, MPS): ~15 min
- **Sequential total: ~45 min**
- **Parallel total: ~30 min** (33% faster)

## Approach

Use `concurrent.futures.ThreadPoolExecutor` to run transcription and diarization in parallel per episode. Threads (not processes) work here because both operations release the GIL during their heavy C/CUDA/Metal computation.

### Data flow change

**Before (sequential):**
```
audio → transcribe() → segments → diarize() → speakers → merge() → output
```

**After (parallel):**
```
audio → ┬─ transcribe() → segments ─┐
         └─ diarize()    → speakers  ─┤→ merge() → output
```

## Steps

### 1. Add `_process_episode_parallel()` to `src/batch_transcribe.py`

New function that runs transcription and diarization concurrently for a single episode:

```python
def _process_episode_parallel(
    ep, audio_dir, model_size, language, engine, initial_prompt,
    pipeline, prefix, duration_limit_minutes=None,
):
    audio_path = str(audio_dir / ep["filename"])
    max_secs = duration_limit_minutes * 60 if duration_limit_minutes else None

    with maybe_truncated_audio(audio_path, max_secs) as effective_audio:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_transcribe = pool.submit(
                transcribe, effective_audio, model_size, language, engine,
                initial_prompt=initial_prompt,
            )
            future_diarize = pool.submit(run_diarization, effective_audio, pipeline)

            result = future_transcribe.result()
            diarization_segments = future_diarize.result()

    _, json_path = save_output(result, audio_path)
    transcript = result["segments"]
    enriched = merge(transcript, diarization_segments)
    _, diarized_json_path = save_diarized_output(enriched, audio_path)

    return json_path, diarized_json_path
```

Key detail: `maybe_truncated_audio` context manager wraps both operations so both use the same (possibly truncated) audio file.

### 2. Modify single-worker loop in `batch_transcribe()`

Replace the sequential transcribe-then-diarize block (lines 268-298) with a call to `_process_episode_parallel()` when diarization is enabled, falling back to the current sequential path otherwise.

### 3. Modify multi-worker diarization pass

In the multi-worker path (lines 300+), after all transcriptions complete, the diarization pass currently runs sequentially. Change this to overlap diarization of episode N with transcription of episode N+1 using a pipeline pattern.

### 4. Update tests

Add test for `_process_episode_parallel` verifying both transcription and diarization results are present in the output.

## Files to Change

| File | Action | Purpose |
|------|--------|---------|
| `src/batch_transcribe.py` | Edit | Add `_process_episode_parallel()`, modify single-worker and multi-worker loops |
| `tests/unit/test_batch_transcribe.py` | Edit | Add tests for parallel execution path |
| `CHANGELOG.md` | Edit | New entry |
| `README.md` | Edit | Note performance improvement |

## Risks

1. **Temp file lifetime** — `maybe_truncated_audio` creates a temp file that must outlive both threads. The `with` block ensures this, but the context manager must not be exited early.
2. **Thread safety of pyannote** — pyannote's `Pipeline.__call__` should be thread-safe for different inputs, but needs verification.
3. **Error propagation** — If transcription fails, diarization result is wasted (acceptable). If diarization fails, transcription result should still be saved (must handle).
4. **Logging interleaving** — Both threads will log concurrently. Consider adding thread-name prefixes.

## Validation

```bash
# Run parallel pipeline on one episode
PYTHONWARNINGS=ignore python src/batch_transcribe.py --language pt --diarize --duration-limit 2 --limit 1

# Verify output exists
ls data/transcripts/*.diarized.json

# Compare wall-clock time with sequential (expect ~33% faster)
# Run tests
pytest tests/unit/test_batch_transcribe.py -v
```
