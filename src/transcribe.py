"""Transcribe audio files to text using OpenAI Whisper or faster-whisper."""

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"

MODEL_SIZES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en", "distil-small.en",
    "medium", "medium.en", "distil-medium.en",
    "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo",
    "turbo", "distil-large-v2", "distil-large-v3",
]

COMPUTE_TYPES = ["float16", "float32", "int8", "int8_float16", "default"]

ENGINES = ["faster-whisper", "openai-whisper", "whisper.cpp"]

WHISPERCPP_MODEL_MAP = {
    "large": "large-v3",
    "turbo": "large-v3-turbo",
}

INITIAL_PROMPT = (
    "Transcrição de podcast brasileiro de cultura pop. "
    "Preserve nomes próprios, termos técnicos e palavras em inglês "
    "exatamente como são falados. Exemplos: Alottoni, Azaghal, "
    "NerdCast, Jovem Nerd, RPG, cosplay, anime, manga. "
    "Ignore músicas e trilhas sonoras de fundo."
)


def _get_openai_whisper_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _transcribe_openai_whisper(audio_path: str, model_size: str, language: str | None, initial_prompt: str | None = None) -> dict:
    import torch
    import whisper

    device = _get_openai_whisper_device()
    logger.info("Loading openai-whisper model '%s' on %s...", model_size, device)
    model = whisper.load_model(model_size, device=device)

    logger.info("Transcribing: %s", audio_path)
    start = time.time()
    result = model.transcribe(audio_path, language=language, verbose=True, initial_prompt=initial_prompt)
    elapsed = time.time() - start
    minutes, secs = divmod(int(elapsed), 60)
    logger.info("Transcription completed in %dm%02ds", minutes, secs)

    return result


def _transcribe_faster_whisper(
    audio_path: str, model_size: str, language: str | None, compute_type: str, initial_prompt: str | None = None
) -> dict:
    from faster_whisper import WhisperModel

    device = "cpu" if platform.system() == "Darwin" else "auto"
    logger.info("Loading faster-whisper model '%s' on %s (compute_type=%s)...", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("Transcribing: %s", audio_path)
    start = time.time()
    segments_iter, info = model.transcribe(audio_path, language=language, vad_filter=True, initial_prompt=initial_prompt)

    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_iter]
    full_text = " ".join(s["text"].strip() for s in segments)

    elapsed = time.time() - start
    minutes, secs = divmod(int(elapsed), 60)
    logger.info("Transcription completed in %dm%02ds (language=%s)", minutes, secs, info.language)

    return {"text": full_text, "segments": segments, "language": info.language}


def _get_audio_duration(audio_path: str) -> float:
    import subprocess

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _format_timestamp(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_whisper_cpp_model(model_size: str = "large", n_threads: int | None = None) -> "Model":
    from pywhispercpp.model import Model

    ggml_model = WHISPERCPP_MODEL_MAP.get(model_size, model_size)
    threads = n_threads or os.cpu_count() or 4
    logger.info("Loading whisper.cpp model '%s' (n_threads=%d)...", ggml_model, threads)
    sys.stderr.flush()

    import tempfile
    tmp = tempfile.TemporaryFile(mode='w+')
    old_stderr_fd = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        model = Model(ggml_model, n_threads=threads, print_progress=False)
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        tmp.seek(0)
        captured = tmp.read()
        tmp.close()

    device = "Metal GPU" if "Metal" in captured else "CPU"
    model_mb = os.path.getsize(model.model_path) / (1024 * 1024) if hasattr(model, 'model_path') else 3094
    logger.info("Model loaded: %.0f MB (device=%s)", model_mb, device)
    return model


def _transcribe_whisper_cpp_with_model(
    model, audio_path: str, language: str | None, initial_prompt: str | None = None,
    log_prefix: str = "",
) -> dict:
    total_duration = _get_audio_duration(audio_path)
    total_str = _format_timestamp(total_duration)

    start = time.time()
    last_logged_pct = -5

    def on_segment(seg):
        nonlocal last_logged_pct
        pos = seg.t1 / 100.0
        pct = int(pos / total_duration * 100) if total_duration > 0 else 0
        if pct - last_logged_pct < 5:
            return
        last_logged_pct = pct
        elapsed = time.time() - start
        if pos > 0:
            eta_secs = elapsed * (total_duration - pos) / pos
            eta_m, eta_s = divmod(int(eta_secs), 60)
            eta_str = f"{eta_m}m{eta_s:02d}s left"
        else:
            eta_str = "estimating..."
        logger.info("%s%s / %s (%d%%) — %s", log_prefix, _format_timestamp(pos), total_str, pct, eta_str)

    transcribe_kwargs = {}
    if language:
        transcribe_kwargs["language"] = language
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    logger.info("%sTranscribing: %s (%s)", log_prefix, audio_path, total_str)
    raw_segments = model.transcribe(audio_path, new_segment_callback=on_segment, **transcribe_kwargs)

    segments = [{"start": s.t0 / 100.0, "end": s.t1 / 100.0, "text": s.text} for s in raw_segments]
    full_text = " ".join(s["text"].strip() for s in segments)

    elapsed = time.time() - start
    minutes, secs = divmod(int(elapsed), 60)
    lang = language or "auto"
    logger.info("%sTranscription completed in %dm%02ds (language=%s)", log_prefix, minutes, secs, lang)

    return {"text": full_text, "segments": segments, "language": lang}


def transcribe_with_model(
    model, audio_path: str, language: str | None = None,
    initial_prompt: str | None = INITIAL_PROMPT, log_prefix: str = "",
) -> dict:
    return _transcribe_whisper_cpp_with_model(model, audio_path, language, initial_prompt, log_prefix)


def _transcribe_whisper_cpp(audio_path: str, model_size: str, language: str | None, initial_prompt: str | None = None) -> dict:
    model = load_whisper_cpp_model(model_size)
    return _transcribe_whisper_cpp_with_model(model, audio_path, language, initial_prompt)


def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
    engine: str = "faster-whisper",
    compute_type: str = "float16",
    initial_prompt: str | None = INITIAL_PROMPT,
) -> dict:
    if engine == "openai-whisper":
        return _transcribe_openai_whisper(audio_path, model_size, language, initial_prompt=initial_prompt)
    if engine == "whisper.cpp":
        return _transcribe_whisper_cpp(audio_path, model_size, language, initial_prompt=initial_prompt)
    return _transcribe_faster_whisper(audio_path, model_size, language, compute_type, initial_prompt=initial_prompt)


def save_output(result: dict, audio_path: str) -> tuple[Path, Path]:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem

    txt_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    txt_path.write_text(result["text"].strip(), encoding="utf-8")
    logger.info("Saved text:     %s", txt_path)

    json_path = TRANSCRIPTS_DIR / f"{stem}.json"
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result["segments"]
    ]
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved segments: %s", json_path)

    return txt_path, json_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, etc.)")
    parser.add_argument("--engine", default="faster-whisper", choices=ENGINES, help="Whisper engine (default: faster-whisper)")
    parser.add_argument("--model", default="base", choices=MODEL_SIZES, help="Whisper model size (default: base)")
    parser.add_argument("--compute-type", default="float16", choices=COMPUTE_TYPES,
                        help="Quantization type for faster-whisper (default: float16, ignored for openai-whisper and whisper.cpp)")
    parser.add_argument("--language", default=None, help="Force language (e.g. 'pt' for Portuguese)")
    parser.add_argument("--initial-prompt", default=INITIAL_PROMPT, help="Initial prompt to guide transcription (default: Portuguese podcast prompt)")
    parser.add_argument("--no-prompt", action="store_true", help="Disable initial prompt")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        logger.error("File not found: %s", args.audio)
        raise SystemExit(1)

    prompt = None if args.no_prompt else args.initial_prompt
    result = transcribe(args.audio, args.model, args.language, args.engine, args.compute_type, initial_prompt=prompt)
    save_output(result, args.audio)
    logger.info("Done!")


if __name__ == "__main__":
    main()
