# Feature Plan: Speaker Diarization

> Identify who is speaking at each moment in podcast transcripts.

**Status:** Planned (not yet implemented)
**Phase:** 3 — Speaker Diarization (from MISSION_ROADMAP.md)
**Created:** 2026-04-24

---

## Key Finding: Whisper Cannot Diarize

**OpenAI Whisper is a speech-to-text model only.** It has no concept of speaker identity — it produces `{start, end, text}` segments but never identifies *who* is speaking. A separate diarization model is required.

---

## Chosen Approach: Modular Post-processing with pyannote.audio

Instead of the WhisperX integrated pipeline originally planned in MISSION_ROADMAP.md, use a **modular post-processing approach**:

```
Audio File (MP3)
       │
       ├──► [src/transcribe.py]  → transcript: [{start, end, text}, ...]
       │
       └──► [src/diarize.py]     → diarization: [{start, end, speaker}, ...]
                    │
                    ▼
              [merge step]       → enriched: [{start, end, text, speaker}, ...]
```

1. **Transcribe** with the existing Whisper pipeline (no changes to `src/transcribe.py`)
2. **Diarize** the same audio with pyannote.audio → produces `(start, end, speaker)` segments
3. **Merge** transcript segments with diarization segments by timestamp overlap

### Why Modular Over WhisperX

- WhisperX has known dependency conflicts on Apple Silicon (ctranslate2 + MPS)
- Keeps transcription and diarization independently upgradeable
- pyannote.audio is actively maintained with well-documented APIs
- Avoids replacing the existing working transcriber

### Why Post-processing Over Pre-processing

- Pre-processing (diarize → split audio → transcribe per speaker) is slower and degrades Whisper quality at segment boundaries
- Whisper performs better on longer continuous audio with surrounding context

---

## Output Format

Current (`src/transcribe.py` output):
```json
[{"start": 0.0, "end": 7.0, "text": "Você está ouvindo Nerdcast..."}]
```

Enriched (after diarization):
```json
[{"start": 0.0, "end": 7.0, "text": "Você está ouvindo Nerdcast...", "speaker": "SPEAKER_00"}]
```

Backward-compatible — the `speaker` field is simply added to each existing segment.

---

## Prerequisites

### HuggingFace Token Setup

pyannote.audio models require accepting licenses on HuggingFace and providing an access token:

1. Create a free account at https://huggingface.co
2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept the model license at https://huggingface.co/pyannote/segmentation-3.0
4. Generate an access token at https://huggingface.co/settings/tokens (select "Read" scope)
5. Add `HF_TOKEN=hf_your_token_here` to the project `.env` file (already gitignored)

### Dependencies

Already listed in `requirements.txt`:
- `pyannote.audio>=3.3`

To remove (not using integrated approach):
- `whisperx>=3.1`

---

## Implementation Details

### New File: `src/diarize.py`

Functions to implement:
- `get_device()` — MPS > CUDA > CPU detection (same pattern as `transcribe.py:14-19`)
- `load_pipeline(hf_token, device)` — load `pyannote/speaker-diarization-3.1`, move to device
- `diarize(audio_path, pipeline)` — run diarization, return `[{start, end, speaker}]`
- `load_transcript(json_path)` — load existing Phase 2 JSON transcript
- `merge(transcript, diarization)` — assign speaker to each segment by max timestamp overlap
- `save_output(segments, audio_path)` — save enriched JSON + speaker-annotated TXT
- `main()` — CLI entry point

### CLI Usage

```bash
# Diarize with auto-detected transcript (looks for data/transcripts/{stem}.json)
python src/diarize.py "data/audio/NerdCast 1025 - Devoradores de Estrelas.mp3"

# Diarize with explicit transcript path
python src/diarize.py "data/audio/episode.mp3" --transcript "data/transcripts/episode.json"

# Force CPU (if MPS causes issues)
python src/diarize.py "data/audio/episode.mp3" --device cpu
```

### Merge Algorithm

```
for each transcript segment:
    find all diarization segments that overlap in time
    compute overlap duration with each
    assign the speaker with the greatest total overlap
    if no overlap found → speaker = "UNKNOWN"
```

O(N*M) complexity where N = transcript segments, M = diarization segments. Negligible for podcast-scale data.

### Tests: `tests/test_diarize.py`

- Test merge algorithm with synthetic segment data (no model loading required)
- Test auto-detection of transcript path from audio filename
- Test missing HF_TOKEN raises clear error with setup instructions

---

## Apple Silicon Considerations

- **pyannote.audio on MPS:** Segmentation and embedding models are PyTorch-based and support `device="mps"`. Some operations may fall back to CPU automatically.
- **Memory:** Diarization pipeline loads ~110MB of models. Since this runs *after* transcription (not simultaneously), Whisper can be unloaded first.
- **Performance:** Expect 10-30x realtime on Apple Silicon (90-min episode → 3-9 min for diarization).
- **Fallback:** If MPS causes issues, the code should gracefully fall back to CPU with a warning.

---

## Files to Change

| File | Action |
|---|---|
| `src/diarize.py` | **Create** — core diarization + merge module |
| `src/transcribe.py` | No changes |
| `requirements.txt` | Remove `whisperx>=3.1` line |
| `.env` | User adds `HF_TOKEN` (gitignored) |
| `tests/test_diarize.py` | **Create** — merge algorithm unit tests |

---

## Verification Checklist

- [ ] `pip install pyannote.audio>=3.3` succeeds
- [ ] `HF_TOKEN` set in `.env`
- [ ] Run on existing episode produces enriched JSON with `speaker` field
- [ ] At least 2 distinct speakers detected in test episode
- [ ] MPS acceleration works (or graceful CPU fallback with warning)
- [ ] Existing transcript format is preserved (new field is additive)

---

## Future Enhancements (Phase 3 stretch goals)

- **Cross-episode speaker re-identification:** Extract voice embeddings, cluster across episodes, map to real names (Jovem Nerd, Azaghal, etc.)
- **LLM-assisted identification:** Use local LLM to infer names from conversational context and episode descriptions
- **Speaker registry:** Manual labeling maps cluster IDs to real names; new episodes auto-match
