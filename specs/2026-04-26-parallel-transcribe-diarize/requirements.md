# Requirements: Parallel Transcription + Diarization

## Functional Requirements

1. **Concurrent execution** — When `--diarize` is enabled, transcription and diarization run concurrently per episode using `ThreadPoolExecutor` or `asyncio`, then merge when both complete.
2. **Same output** — Output format (`.diarized.json`, `.diarized.txt`) is identical to the current sequential pipeline.
3. **Fallback to sequential** — If concurrency fails or a flag `--no-parallel-diarize` is passed, fall back to the current sequential behavior.
4. **Duration-limit support** — Both transcription and diarization respect `--duration-limit` when running in parallel.
5. **Backwards compatible** — Existing CLI behavior is unchanged. Parallel diarization is the new default when `--diarize` is used, with opt-out available.

## Non-Functional Requirements

1. **Wall-clock speedup** — For a single episode, total time should approach `max(transcription_time, diarization_time)` instead of `transcription_time + diarization_time`.
2. **Memory** — Both models coexist in memory already (Whisper ~3GB + pyannote ~110MB on 32GB system). No additional memory overhead from parallelism.
3. **Device contention** — Transcription (CPU/int8 for faster-whisper) and diarization (MPS/GPU for pyannote) use different devices, so contention is minimal on Apple Silicon.
4. **Error isolation** — A diarization failure does not cancel or corrupt the transcription result, and vice versa.

## Constraints

- Python 3.12+, Apple Silicon (MPS/Metal)
- `faster-whisper` runs on CPU (int8); pyannote runs on MPS — naturally parallel on Apple Silicon
- Multi-worker transcription mode (whisper.cpp) already uses `ProcessPoolExecutor`; parallel diarization applies to the single-worker path and post-worker diarization pass
