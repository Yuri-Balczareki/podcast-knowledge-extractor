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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = [
    _PROJECT_ROOT / "tests" / "functional" / "fixtures" / "nerdcast_1025_clip.mp3",
    _PROJECT_ROOT / "tests" / "functional" / "fixtures" / "nerdcast_1026_clip.mp3",
    _PROJECT_ROOT / "tests" / "functional" / "fixtures" / "nerdcast_1027_clip.mp3",
]


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
    try:
        model = whisper.load_model(model_size, device=device)
    except NotImplementedError:
        if device == "mps":
            logger.warning("MPS failed for model '%s', falling back to CPU", model_size)
            device = "cpu"
            model = whisper.load_model(model_size, device=device)
        else:
            raise

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


def save_markdown_report(
    results: list[TranscriptionResult],
    metrics: dict[str, EngineMetrics],
    pairwise: list[PairwiseComparison],
    audio_duration_s: float,
    model_size: str,
    language: str | None,
    initial_prompt: str | None,
    output_dir: Path,
    stem: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dur_min = audio_duration_s / 60
    lang_str = language if language else "auto-detect"
    prompt_str = (
        f"{initial_prompt[:60]}..." if initial_prompt and len(initial_prompt) > 60
        else (initial_prompt if initial_prompt else "none")
    )
    engines = [r.engine for r in results]

    lines = [
        "# Whisper Transcription Comparison",
        "",
        "## Metadata",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Audio duration | {dur_min:.1f} min ({audio_duration_s:.0f}s) |",
        f"| Model size | {model_size} |",
        f"| Language | {lang_str} |",
        f"| Initial prompt | {prompt_str} |",
        "",
        "## Performance",
        "",
        "| Metric | " + " | ".join(engines) + " |",
        "|--------" + "".join("|-" * len(engines)) + "|",
    ]

    lines.append("| Device | " + " | ".join(r.device_used for r in results) + " |")
    lines.append("| Wall time | " + " | ".join(format_time(r.wall_time_s) for r in results) + " |")
    lines.append(
        "| Real-time factor | "
        + " | ".join(f"{audio_duration_s / r.wall_time_s:.1f}x" if r.wall_time_s else "N/A" for r in results)
        + " |"
    )
    lines.append(
        "| Memory delta (RSS) | " + " | ".join(f"{r.memory_rss_mb:.1f} MB" for r in results) + " |"
    )

    lines += [
        "",
        "## Output Quality",
        "",
        "| Metric | " + " | ".join(engines) + " |",
        "|--------" + "".join("|-" * len(engines)) + "|",
    ]
    lines.append("| Word count | " + " | ".join(f"{metrics[e].word_count:,d}" for e in engines) + " |")
    lines.append("| Character count | " + " | ".join(f"{metrics[e].char_count:,d}" for e in engines) + " |")
    lines.append("| Segment count | " + " | ".join(f"{metrics[e].segment_count:,d}" for e in engines) + " |")
    lines.append(
        "| Avg segment duration | "
        + " | ".join(f"{metrics[e].avg_segment_duration:.2f}s" for e in engines)
        + " |"
    )

    if pairwise:
        lines += [
            "",
            "## Pairwise Comparison",
            "",
            "| Pair | Word diff | WER |",
            "|------|-----------|-----|",
        ]
        for p in pairwise:
            sign = "+" if p.word_count_diff >= 0 else ""
            lines.append(
                f"| {p.engine_a} vs {p.engine_b} "
                f"| {sign}{p.word_count_diff} ({sign}{p.word_count_diff_pct:.1f}%) "
                f"| {p.wer:.4f} ({p.wer * 100:.2f}%) |"
            )

    lines.append("")

    path = output_dir / f"{stem}_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _get_audio_duration_ffprobe(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def save_benchmark_summary(
    all_file_data: list[dict],
    engines: list[str],
    model_size: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(all_file_data)

    device_used = {}
    avg_wall = {}
    avg_rtf = {}
    avg_mem = {}
    avg_words = {}
    for engine in engines:
        wall_times, rtfs, mems, words = [], [], [], []
        for fd in all_file_data:
            r = next(r for r in fd["results"] if r.engine == engine)
            wall_times.append(r.wall_time_s)
            rtfs.append(fd["audio_duration_s"] / r.wall_time_s if r.wall_time_s else 0)
            mems.append(r.memory_rss_mb)
            words.append(fd["metrics"][engine].word_count)
        device_used[engine] = next(r for r in all_file_data[0]["results"] if r.engine == engine).device_used
        avg_wall[engine] = mean(wall_times)
        avg_rtf[engine] = mean(rtfs)
        avg_mem[engine] = mean(mems)
        avg_words[engine] = mean(words)

    pair_wers: dict[tuple[str, str], list[float]] = {}
    for fd in all_file_data:
        for p in fd["pairwise"]:
            pair_wers.setdefault((p.engine_a, p.engine_b), []).append(p.wer)

    lines = [
        f"# Benchmark Summary ({n} files)",
        "",
        "## Files",
        "",
        "| # | File |",
        "|---|------|",
    ]
    for i, fd in enumerate(all_file_data, 1):
        lines.append(f"| {i} | {fd['stem']}.mp3 |")

    lines += [
        "",
        f"## Averaged Performance (model: {model_size})",
        "",
        "| Metric | " + " | ".join(engines) + " |",
        "|--------" + "".join("|-" * len(engines)) + "|",
        "| Device | " + " | ".join(device_used[e] for e in engines) + " |",
        "| Avg wall time | " + " | ".join(format_time(avg_wall[e]) for e in engines) + " |",
        "| Avg real-time factor | " + " | ".join(f"{avg_rtf[e]:.1f}x" for e in engines) + " |",
        "| Avg memory delta (RSS) | " + " | ".join(f"{avg_mem[e]:.1f} MB" for e in engines) + " |",
        "",
        "## Averaged Output Quality",
        "",
        "| Metric | " + " | ".join(engines) + " |",
        "|--------" + "".join("|-" * len(engines)) + "|",
        "| Avg word count | " + " | ".join(f"{avg_words[e]:.0f}" for e in engines) + " |",
    ]

    if pair_wers:
        lines += [
            "",
            "## Averaged Pairwise WER",
            "",
            "| Pair | Avg WER |",
            "|------|---------|",
        ]
        for (ea, eb), wers in pair_wers.items():
            avg = mean(wers)
            lines.append(f"| {ea} vs {eb} | {avg:.4f} ({avg * 100:.2f}%) |")

    lines.append("")

    path = output_dir / f"benchmark_summary_{model_size}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_single_file(
    audio_path: Path,
    model_size: str,
    language: str | None,
    prompt: str | None,
    engine_names: list[str],
    max_duration: int | None,
    output_dir: Path,
    output_format: str | None,
) -> dict:
    if not audio_path.exists():
        logger.error("File not found: %s", audio_path)
        raise SystemExit(1)

    needs_openai_audio = any(e in engine_names for e in ("openai-whisper", "faster-whisper"))
    audio_array = None

    if needs_openai_audio or max_duration:
        import whisper

        logger.info("Loading audio: %s", audio_path)
        audio_array = whisper.load_audio(str(audio_path))

        if max_duration:
            max_samples = max_duration * SAMPLE_RATE
            audio_array = audio_array[:max_samples]
            logger.info("Truncated audio to first %ds", max_duration)

        audio_duration_s = len(audio_array) / SAMPLE_RATE
    else:
        audio_duration_s = _get_audio_duration_ffprobe(str(audio_path))

    logger.info("Audio duration: %.1f min (%.0fs)", audio_duration_s / 60, audio_duration_s)
    logger.info("Model: %s | Language: %s", model_size, language or "auto-detect")

    results: list[TranscriptionResult] = []
    for engine_name in engine_names:
        runner = ENGINE_RUNNERS[engine_name]
        result = runner(str(audio_path), model_size, language, initial_prompt=prompt, audio_array=audio_array)
        results.append(result)

    metrics = {r.engine: compute_engine_metrics(r) for r in results}
    pairwise = compute_pairwise_comparisons(results)
    table = format_comparison_table(results, metrics, pairwise, audio_duration_s, model_size, language, initial_prompt=prompt)
    logger.info(table)

    stem = audio_path.stem
    saved_files = save_transcripts(results, output_dir, stem)
    for f in saved_files:
        logger.info("Saved: %s", f)

    if output_format in ("json", "all"):
        json_path = save_metrics_json(results, metrics, pairwise, audio_duration_s, model_size, output_dir, stem)
        logger.info("Saved metrics: %s", json_path)

    if output_format in ("markdown", "all"):
        md_path = save_markdown_report(
            results, metrics, pairwise, audio_duration_s, model_size, language, prompt, output_dir, stem,
        )
        logger.info("Saved report: %s", md_path)

    return {
        "stem": stem,
        "audio_duration_s": audio_duration_s,
        "results": results,
        "metrics": metrics,
        "pairwise": pairwise,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compare Whisper transcription engines")
    parser.add_argument("audio", nargs="*", help="Audio file(s). Defaults to fixture clips if omitted.")
    parser.add_argument("--model", default="base", choices=MODEL_CHOICES, help="Whisper model size (default: base)")
    parser.add_argument("--language", default="pt", help="Language code (default: pt for Portuguese)")
    parser.add_argument("--max-duration", type=int, default=None, help="Limit audio to first N seconds (for quick tests)")
    parser.add_argument("--initial-prompt", default=INITIAL_PROMPT, help="Initial prompt to guide transcription")
    parser.add_argument("--no-prompt", action="store_true", help="Disable initial prompt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output files")
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "all"],
        default=None,
        help="Save comparison report to file (json, markdown, or all)",
    )
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINE_RUNNERS.keys()),
        choices=list(ENGINE_RUNNERS.keys()), help="Engines to compare (default: all)",
    )
    parser.add_argument("--parallel", action="store_true", help="Run files in parallel (one process per file)")
    args = parser.parse_args()

    audio_paths = [Path(a) for a in args.audio] if args.audio else DEFAULT_FIXTURES
    prompt = None if args.no_prompt else args.initial_prompt
    output_dir = args.output_dir / args.model

    logger.info("Benchmarking %d file(s)%s", len(audio_paths), " (parallel)" if args.parallel else "")

    run_fn = partial(
        run_single_file,
        model_size=args.model, language=args.language, prompt=prompt,
        engine_names=args.engines, max_duration=args.max_duration,
        output_dir=output_dir, output_format=args.output_format,
    )

    if args.parallel and len(audio_paths) > 1:
        with ProcessPoolExecutor(max_workers=len(audio_paths)) as pool:
            all_file_data = list(pool.map(run_fn, audio_paths))
    else:
        all_file_data = [run_fn(p) for p in audio_paths]

    if len(all_file_data) > 1 and args.output_format in ("markdown", "all"):
        summary_path = save_benchmark_summary(all_file_data, args.engines, args.model, output_dir)
        logger.info("Saved summary: %s", summary_path)

    logger.info("Comparison complete.")


if __name__ == "__main__":
    main()
