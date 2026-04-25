"""Compare transcription engines (OpenAI Whisper, faster-whisper, whisper.cpp) on the same audio file."""

import argparse
import gc
import itertools
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import jiwer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "comparisons"
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]

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


@dataclass
class TranscriptionResult:
    engine: str
    full_text: str
    segments: list[dict]
    wall_time_s: float
    memory_rss_mb: float
    device_used: str


@dataclass
class EngineMetrics:
    word_count: int
    char_count: int
    segment_count: int
    avg_segment_duration: float


@dataclass
class PairwiseComparison:
    engine_a: str
    engine_b: str
    word_count_diff: int
    word_count_diff_pct: float
    char_count_diff: int
    char_count_diff_pct: float
    wer: float


def get_current_rss_mb() -> float:
    if sys.platform == "darwin":
        try:
            output = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
            return int(output) / 1024
        except (ValueError, OSError):
            return 0.0
    else:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except (FileNotFoundError, ValueError):
            return 0.0
    return 0.0


def run_openai_whisper(
    audio_path: str, model_size: str, language: str | None, initial_prompt: str | None = None, audio_array=None
) -> TranscriptionResult:
    import torch
    import whisper

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("OpenAI Whisper — loading model '%s' on %s...", model_size, device)

    rss_before = get_current_rss_mb()
    model = whisper.load_model(model_size, device=device)

    audio_input = audio_array if audio_array is not None else audio_path
    start = time.perf_counter()
    result = model.transcribe(audio_input, language=language, verbose=False, initial_prompt=initial_prompt)
    elapsed = time.perf_counter() - start
    rss_after = get_current_rss_mb()

    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]
    full_text = result["text"].strip()

    logger.info("OpenAI Whisper — done in %.1fs", elapsed)

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return TranscriptionResult(
        engine="openai-whisper",
        full_text=full_text,
        segments=segments,
        wall_time_s=elapsed,
        memory_rss_mb=max(rss_after - rss_before, 0.0),
        device_used=device,
    )


def run_faster_whisper(
    audio_path: str, model_size: str, language: str | None, initial_prompt: str | None = None, audio_array=None
) -> TranscriptionResult:
    from faster_whisper import WhisperModel

    gc.collect()

    device = "cuda"
    compute_type = "float16"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "float32"
    except ImportError:
        device = "cpu"
        compute_type = "float32"

    logger.info("faster-whisper — loading model '%s' on %s (%s)...", model_size, device, compute_type)

    rss_before = get_current_rss_mb()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    audio_input = audio_array if audio_array is not None else audio_path
    start = time.perf_counter()
    segments_gen, _info = model.transcribe(audio_input, language=language, log_progress=False, initial_prompt=initial_prompt)
    segments_raw = list(segments_gen)
    elapsed = time.perf_counter() - start
    rss_after = get_current_rss_mb()

    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments_raw]
    full_text = " ".join(s["text"] for s in segments)

    logger.info("faster-whisper — done in %.1fs", elapsed)

    del model
    gc.collect()

    return TranscriptionResult(
        engine="faster-whisper",
        full_text=full_text,
        segments=segments,
        wall_time_s=elapsed,
        memory_rss_mb=max(rss_after - rss_before, 0.0),
        device_used=device,
    )


def run_whisper_cpp(
    audio_path: str, model_size: str, language: str | None, initial_prompt: str | None = None, audio_array=None
) -> TranscriptionResult:
    from pywhispercpp.model import Model

    gc.collect()

    resolved_model = WHISPERCPP_MODEL_MAP.get(model_size, model_size)

    n_threads = os.cpu_count() or 4
    device = "metal" if platform.system() == "Darwin" else "cpu"
    logger.info("whisper.cpp — loading model '%s' (%s, n_threads=%d)...", resolved_model, device, n_threads)

    rss_before = get_current_rss_mb()
    model = Model(resolved_model, n_threads=n_threads)

    transcribe_kwargs = {}
    if language:
        transcribe_kwargs["language"] = language
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    start = time.perf_counter()
    audio_input = audio_array if audio_array is not None else audio_path
    raw_segments = model.transcribe(audio_input, **transcribe_kwargs)
    elapsed = time.perf_counter() - start
    rss_after = get_current_rss_mb()

    segments = [{"start": s.t0 / 100.0, "end": s.t1 / 100.0, "text": s.text.strip()} for s in raw_segments]
    full_text = " ".join(s["text"] for s in segments)

    logger.info("whisper.cpp — done in %.1fs", elapsed)

    del model
    gc.collect()

    return TranscriptionResult(
        engine="whisper.cpp",
        full_text=full_text,
        segments=segments,
        wall_time_s=elapsed,
        memory_rss_mb=max(rss_after - rss_before, 0.0),
        device_used=device,
    )


ENGINE_RUNNERS = {
    "openai-whisper": run_openai_whisper,
    "faster-whisper": run_faster_whisper,
    "whisper.cpp": run_whisper_cpp,
}


def compute_engine_metrics(result: TranscriptionResult) -> EngineMetrics:
    words = result.full_text.split()
    avg_dur = mean(s["end"] - s["start"] for s in result.segments) if result.segments else 0.0
    return EngineMetrics(
        word_count=len(words),
        char_count=len(result.full_text),
        segment_count=len(result.segments),
        avg_segment_duration=avg_dur,
    )


def compute_pairwise_comparisons(results: list[TranscriptionResult]) -> list[PairwiseComparison]:
    comparisons = []
    for a, b in itertools.combinations(results, 2):
        wc_a = len(a.full_text.split())
        wc_b = len(b.full_text.split())
        wc_diff = wc_b - wc_a
        wc_diff_pct = (wc_diff / wc_a * 100) if wc_a else 0.0

        cc_a = len(a.full_text)
        cc_b = len(b.full_text)
        cc_diff = cc_b - cc_a
        cc_diff_pct = (cc_diff / cc_a * 100) if cc_a else 0.0

        wer = jiwer.wer(a.full_text, b.full_text) if a.full_text else 0.0

        comparisons.append(PairwiseComparison(
            engine_a=a.engine,
            engine_b=b.engine,
            word_count_diff=wc_diff,
            word_count_diff_pct=wc_diff_pct,
            char_count_diff=cc_diff,
            char_count_diff_pct=cc_diff_pct,
            wer=wer,
        ))
    return comparisons


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _sanitize_engine_name(engine: str) -> str:
    return engine.replace("-", "_").replace(".", "_")


def format_comparison_table(
    results: list[TranscriptionResult],
    metrics: dict[str, EngineMetrics],
    pairwise: list[PairwiseComparison],
    audio_duration_s: float,
    model_size: str,
    language: str | None,
    initial_prompt: str | None = None,
) -> str:
    col_w = max(20, max(len(r.engine) for r in results) + 4)
    dur_min = audio_duration_s / 60
    lang_str = language if language else "auto-detect"
    prompt_str = (
        f'"{initial_prompt[:60]}..."' if initial_prompt and len(initial_prompt) > 60
        else (f'"{initial_prompt}"' if initial_prompt else "none")
    )

    lines = [
        "",
        "=" * (26 + col_w * len(results)),
        "            Whisper Transcription Comparison",
        "=" * (26 + col_w * len(results)),
        "",
        f"  Audio duration:    {dur_min:.1f} min ({audio_duration_s:.0f}s)",
        f"  Model size:        {model_size}",
        f"  Language:          {lang_str}",
        f"  Initial prompt:    {prompt_str}",
        "",
        "  --- Performance ---",
    ]

    header = f"  {'':24s}" + "".join(f"{r.engine:>{col_w}s}" for r in results)
    lines.append(header)

    row = f"  {'Device:':<24s}" + "".join(f"{r.device_used:>{col_w}s}" for r in results)
    lines.append(row)

    row = f"  {'Wall time:':<24s}" + "".join(f"{format_time(r.wall_time_s):>{col_w}s}" for r in results)
    lines.append(row)

    row = f"  {'Real-time factor:':<24s}" + "".join(
        f"{audio_duration_s / r.wall_time_s if r.wall_time_s else 0.0:>{col_w - 1}.1f}x" for r in results
    )
    lines.append(row)

    row = f"  {'Memory delta (RSS):':<24s}" + "".join(
        f"{r.memory_rss_mb:>{col_w - 3}.1f} MB" for r in results
    )
    lines.append(row)

    lines.append("")
    lines.append("  --- Output Quality ---")

    header = f"  {'':24s}" + "".join(f"{r.engine:>{col_w}s}" for r in results)
    lines.append(header)

    row = f"  {'Word count:':<24s}" + "".join(f"{metrics[r.engine].word_count:>{col_w},d}" for r in results)
    lines.append(row)

    row = f"  {'Character count:':<24s}" + "".join(f"{metrics[r.engine].char_count:>{col_w},d}" for r in results)
    lines.append(row)

    row = f"  {'Segment count:':<24s}" + "".join(f"{metrics[r.engine].segment_count:>{col_w},d}" for r in results)
    lines.append(row)

    row = f"  {'Avg segment duration:':<24s}" + "".join(
        f"{metrics[r.engine].avg_segment_duration:>{col_w - 1}.2f}s" for r in results
    )
    lines.append(row)

    if pairwise:
        lines.append("")
        lines.append("  --- Pairwise Comparison ---")
        for p in pairwise:
            sign = "+" if p.word_count_diff >= 0 else ""
            lines.append(f"  {p.engine_a} vs {p.engine_b}:")
            lines.append(f"    Word diff: {sign}{p.word_count_diff} ({sign}{p.word_count_diff_pct:.1f}%)")
            lines.append(f"    WER:       {p.wer:.4f} ({p.wer * 100:.2f}%)")

    lines.append("")
    lines.append("=" * (26 + col_w * len(results)))
    lines.append("")
    return "\n".join(lines)


def save_transcripts(results: list[TranscriptionResult], output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for result in results:
        tag = _sanitize_engine_name(result.engine)

        txt_path = output_dir / f"{stem}_{tag}.txt"
        txt_path.write_text(result.full_text, encoding="utf-8")
        saved.append(txt_path)

        json_path = output_dir / f"{stem}_{tag}.json"
        json_path.write_text(json.dumps(result.segments, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(json_path)

    return saved


def save_metrics_json(
    results: list[TranscriptionResult],
    metrics: dict[str, EngineMetrics],
    pairwise: list[PairwiseComparison],
    audio_duration_s: float,
    model_size: str,
    output_dir: Path,
    stem: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    engines_data = {}
    for r in results:
        m = metrics[r.engine]
        engines_data[_sanitize_engine_name(r.engine)] = {
            "device": r.device_used,
            "wall_time_s": r.wall_time_s,
            "rtf": audio_duration_s / r.wall_time_s if r.wall_time_s else 0,
            "memory_rss_delta_mb": r.memory_rss_mb,
            "word_count": m.word_count,
            "char_count": m.char_count,
            "segment_count": m.segment_count,
            "avg_segment_duration_s": m.avg_segment_duration,
        }

    pairwise_data = [
        {
            "engines": [p.engine_a, p.engine_b],
            "word_count_diff": p.word_count_diff,
            "word_count_diff_pct": p.word_count_diff_pct,
            "char_count_diff": p.char_count_diff,
            "char_count_diff_pct": p.char_count_diff_pct,
            "wer": p.wer,
        }
        for p in pairwise
    ]

    data = {
        "audio_duration_s": audio_duration_s,
        "model_size": model_size,
        "engines": engines_data,
        "pairwise_comparisons": pairwise_data,
    }

    path = output_dir / f"{stem}_comparison.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _get_audio_duration_ffprobe(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compare Whisper transcription engines")
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, etc.)")
    parser.add_argument("--model", default="base", choices=MODEL_CHOICES, help="Whisper model size (default: base)")
    parser.add_argument("--language", default=None, help="Force language code (e.g. 'pt' for Portuguese)")
    parser.add_argument("--max-duration", type=int, default=None, help="Limit audio to first N seconds (for quick tests)")
    parser.add_argument("--initial-prompt", default=INITIAL_PROMPT, help="Initial prompt to guide transcription")
    parser.add_argument("--no-prompt", action="store_true", help="Disable initial prompt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output files")
    parser.add_argument("--save-json", action="store_true", help="Save metrics to a JSON file")
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINE_RUNNERS.keys()),
        choices=list(ENGINE_RUNNERS.keys()), help="Engines to compare (default: all)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("File not found: %s", audio_path)
        raise SystemExit(1)

    needs_openai_audio = any(e in args.engines for e in ("openai-whisper", "faster-whisper"))
    audio_array = None

    if needs_openai_audio or args.max_duration:
        import whisper

        logger.info("Loading audio: %s", audio_path)
        audio_array = whisper.load_audio(str(audio_path))

        if args.max_duration:
            max_samples = args.max_duration * SAMPLE_RATE
            audio_array = audio_array[:max_samples]
            logger.info("Truncated audio to first %ds", args.max_duration)

        audio_duration_s = len(audio_array) / SAMPLE_RATE
    else:
        audio_duration_s = _get_audio_duration_ffprobe(str(audio_path))

    logger.info("Audio duration: %.1f min (%.0fs)", audio_duration_s / 60, audio_duration_s)
    logger.info("Model: %s | Language: %s", args.model, args.language or "auto-detect")

    prompt = None if args.no_prompt else args.initial_prompt

    results: list[TranscriptionResult] = []
    for engine_name in args.engines:
        runner = ENGINE_RUNNERS[engine_name]
        result = runner(str(audio_path), args.model, args.language, initial_prompt=prompt, audio_array=audio_array)
        results.append(result)

    metrics = {r.engine: compute_engine_metrics(r) for r in results}
    pairwise = compute_pairwise_comparisons(results)
    table = format_comparison_table(results, metrics, pairwise, audio_duration_s, args.model, args.language, initial_prompt=prompt)
    logger.info(table)

    stem = audio_path.stem
    saved_files = save_transcripts(results, args.output_dir, stem)
    for f in saved_files:
        logger.info("Saved: %s", f)

    if args.save_json:
        json_path = save_metrics_json(results, metrics, pairwise, audio_duration_s, args.model, args.output_dir, stem)
        logger.info("Saved metrics: %s", json_path)

    logger.info("Comparison complete.")


if __name__ == "__main__":
    main()
