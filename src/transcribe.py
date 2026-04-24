"""Transcribe audio files to text using OpenAI Whisper."""

import argparse
import json
import time
from pathlib import Path

import torch
import whisper

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transcribe(audio_path: str, model_size: str = "base", language: str | None = None) -> dict:
    device = get_device()
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing: {audio_path}")
    start = time.time()

    result = model.transcribe(audio_path, language=language, verbose=True)

    elapsed = time.time() - start
    minutes, secs = divmod(int(elapsed), 60)
    print(f"\nTranscription completed in {minutes}m{secs:02d}s")

    return result


def save_output(result: dict, audio_path: str) -> tuple[Path, Path]:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem

    txt_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    txt_path.write_text(result["text"].strip(), encoding="utf-8")
    print(f"Saved text:     {txt_path}")

    json_path = TRANSCRIPTS_DIR / f"{stem}.json"
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result["segments"]
    ]
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved segments: {json_path}")

    return txt_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, etc.)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None, help="Force language (e.g. 'pt' for Portuguese)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"File not found: {args.audio}")
        raise SystemExit(1)

    result = transcribe(args.audio, args.model, args.language)
    save_output(result, args.audio)
    print("\nDone!")


if __name__ == "__main__":
    main()
