"""Speaker diarization using pyannote.audio with transcript merging."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"


def get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(hf_token: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    logger.info("Loading pyannote speaker-diarization-3.1 on %s...", device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    try:
        pipeline.to(torch.device(device))
    except RuntimeError:
        logger.warning("Failed to use %s, falling back to CPU", device)
        pipeline.to(torch.device("cpu"))
    return pipeline


def diarize(audio_path: str, pipeline: Pipeline) -> list[dict]:
    logger.info("Diarizing: %s", audio_path)
    start = time.time()

    annotation = pipeline(audio_path)

    elapsed = time.time() - start
    minutes, secs = divmod(int(elapsed), 60)
    logger.info("Diarization completed in %dm%02ds", minutes, secs)

    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": segment.start, "end": segment.end, "speaker": speaker})

    speakers = {s["speaker"] for s in segments}
    logger.info("Detected %d speakers: %s", len(speakers), ", ".join(sorted(speakers)))

    return segments


def load_transcript(json_path: str | Path) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge(transcript: list[dict], diarization: list[dict]) -> list[dict]:
    enriched = []
    for seg in transcript:
        speaker_overlap: dict[str, float] = defaultdict(float)
        for d in diarization:
            overlap = max(0.0, min(seg["end"], d["end"]) - max(seg["start"], d["start"]))
            if overlap > 0:
                speaker_overlap[d["speaker"]] += overlap

        speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else "UNKNOWN"
        enriched.append({**seg, "speaker": speaker})

    return enriched


def save_output(segments: list[dict], audio_path: str) -> tuple[Path, Path]:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem

    json_path = TRANSCRIPTS_DIR / f"{stem}.diarized.json"
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved diarized JSON: %s", json_path)

    txt_path = TRANSCRIPTS_DIR / f"{stem}.diarized.txt"
    lines = [f"[{s['speaker']}] {s['text'].strip()}" for s in segments]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved diarized text: %s", txt_path)

    return txt_path, json_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Speaker diarization with pyannote.audio")
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, etc.)")
    parser.add_argument("--transcript", default=None, help="Path to transcript JSON (default: auto-detect)")
    parser.add_argument("--device", default=None, help="Force device: cpu, cuda, or mps (default: auto-detect)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("File not found: %s", audio_path)
        raise SystemExit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error(
            "HF_TOKEN not set. To configure:\n"
            "  1. Create account at https://huggingface.co\n"
            "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. Accept license at https://huggingface.co/pyannote/segmentation-3.0\n"
            "  4. Generate token at https://huggingface.co/settings/tokens\n"
            "  5. Run: export HF_TOKEN=hf_your_token_here"
        )
        raise SystemExit(1)

    transcript_path = Path(args.transcript) if args.transcript else TRANSCRIPTS_DIR / f"{audio_path.stem}.json"
    if not transcript_path.exists():
        logger.error("Transcript not found: %s", transcript_path)
        raise SystemExit(1)

    device = args.device or get_device()
    pipeline = load_pipeline(hf_token, device)
    diarization_segments = diarize(str(audio_path), pipeline)
    transcript = load_transcript(transcript_path)
    enriched = merge(transcript, diarization_segments)
    save_output(enriched, str(audio_path))

    logger.info("Done!")


if __name__ == "__main__":
    main()
