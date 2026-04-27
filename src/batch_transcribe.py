"""Batch transcription and diarization pipeline for podcast episodes.

Reads the episode CSV, identifies downloaded-but-not-transcribed episodes,
and transcribes them using the specified Whisper engine. Supports
parallel workers (whisper.cpp only), speaker diarization via pyannote.audio,
and tracks progress in the CSV.

Usage:
    python src/batch_transcribe.py                              # faster-whisper (default)
    python src/batch_transcribe.py --engine whisper.cpp          # whisper.cpp engine
    python src/batch_transcribe.py --engine whisper.cpp --workers 2  # parallel (whisper.cpp only)
    python src/batch_transcribe.py --limit 10                   # transcribe up to 10 episodes
    python src/batch_transcribe.py --dry-run                    # show what would be transcribed
    python src/batch_transcribe.py --diarize                    # transcribe + diarize in one pass
    python src/batch_transcribe.py --skip-transcription --diarize  # diarize already-transcribed episodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyannote.audio import Pipeline

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.scraper import AUDIO_DIR, EPISODES_CSV, load_csv, save_csv
from src.transcribe import (
    ENGINES,
    INITIAL_PROMPT,
    TRANSCRIPTS_DIR,
    load_whisper_cpp_model,
    maybe_truncated_audio,
    save_output,
    transcribe,
    transcribe_with_model,
)
from src.diarize import (
    diarize as run_diarization,
    get_device,
    load_pipeline,
    load_transcript,
    merge,
    save_output as save_diarized_output,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"

_worker_model = None
_worker_id = None


def _init_worker(counter, model_size: str, n_threads: int) -> None:
    global _worker_model, _worker_id
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info(
        "[W%d] Loading whisper.cpp model '%s' (%d threads)...",
        _worker_id,
        model_size,
        n_threads,
    )
    _worker_model = load_whisper_cpp_model(model_size, n_threads)
    logger.info("[W%d] Model loaded and ready.", _worker_id)


def _transcribe_episode(
    audio_path: str,
    language: str | None,
    initial_prompt: str | None,
    log_prefix: str,
    duration_limit_minutes: float | None = None,
) -> dict:
    full_prefix = f"[W{_worker_id}] {log_prefix}"
    result = transcribe_with_model(
        _worker_model,
        audio_path,
        language,
        initial_prompt,
        log_prefix=full_prefix,
        duration_limit_minutes=duration_limit_minutes,
    )
    json_path, _ = save_output(result, audio_path)
    return {"json_path": str(json_path)}


def detect_existing_transcripts(episodes: list[dict]) -> int:
    """Scan transcript dir for existing files and mark matching episodes as transcribed."""
    detected = 0
    for ep in episodes:
        if ep["status"] != "downloaded" or not ep.get("filename"):
            continue
        if ep.get("transcription_status") == "transcribed":
            continue
        stem = Path(ep["filename"]).stem
        json_path = TRANSCRIPTS_DIR / f"{stem}.json"
        if json_path.exists():
            ep["transcription_status"] = "transcribed"
            ep["transcript_path"] = str(json_path)
            detected += 1
    return detected


def get_pending(episodes: list[dict]) -> list[dict]:
    """Return episodes that are downloaded but not yet transcribed."""
    return [
        ep
        for ep in episodes
        if ep["status"] == "downloaded"
        and ep.get("transcription_status") in ("not_transcribed", "failed")
        and ep.get("filename")
    ]


def get_pending_diarization(episodes: list[dict]) -> list[dict]:
    """Return transcribed episodes that are not yet diarized."""
    return [
        ep
        for ep in episodes
        if ep["status"] == "downloaded"
        and ep.get("transcription_status") == "transcribed"
        and ep.get("transcript_path")
        and ep.get("diarization_status") in ("not_diarized", "failed")
        and ep.get("filename")
    ]


def detect_existing_diarized_transcripts(episodes: list[dict]) -> int:
    """Scan transcript dir for existing diarized files and mark matching episodes."""
    detected = 0
    for ep in episodes:
        if ep["status"] != "downloaded" or not ep.get("filename"):
            continue
        if ep.get("diarization_status") == "diarized":
            continue
        stem = Path(ep["filename"]).stem
        json_path = TRANSCRIPTS_DIR / f"{stem}.diarized.json"
        if json_path.exists():
            ep["diarization_status"] = "diarized"
            ep["diarized_transcript_path"] = str(json_path)
            detected += 1
    return detected


def compute_threads_per_worker(workers: int) -> int:
    """Divide available CPU threads evenly across parallel workers."""
    return max(1, (os.cpu_count() or 4) // workers)


def _diarize_episode(
    ep: dict,
    audio_dir: Path,
    pipeline: "Pipeline",
    duration_limit_minutes: float | None = None,
) -> None:
    audio_path = str(audio_dir / ep["filename"])
    transcript = load_transcript(ep["transcript_path"])
    speakers = ep.get("speakers", "")
    num_speakers = len(speakers.split("|")) if speakers else None
    max_secs = duration_limit_minutes * 60 if duration_limit_minutes else None
    with maybe_truncated_audio(audio_path, max_secs) as effective_audio:
        diarization_segments = run_diarization(
            effective_audio, pipeline,
            min_speakers=num_speakers, max_speakers=num_speakers,
        )
    enriched = merge(transcript, diarization_segments)
    json_path, _ = save_diarized_output(enriched, audio_path)
    ep["diarization_status"] = "diarized"
    ep["diarized_transcript_path"] = str(json_path)


def batch_transcribe(
    csv_path: Path,
    audio_dir: Path,
    model_size: str = "large",
    engine: str = "faster-whisper",
    language: str | None = None,
    workers: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    initial_prompt: str | None = INITIAL_PROMPT,
    enable_diarization: bool = False,
    skip_transcription: bool = False,
    diarization_device: str | None = None,
    duration_limit_minutes: float | None = None,
) -> None:
    """Orchestrate batch transcription and optional diarization for pending episodes."""
    episodes = load_csv(csv_path)

    for ep in episodes:
        ep.setdefault("transcription_status", "not_transcribed")
        ep.setdefault("transcript_path", "")
        ep.setdefault("diarization_status", "not_diarized")
        ep.setdefault("diarized_transcript_path", "")
        ep.setdefault("speakers", "")

    detected = detect_existing_transcripts(episodes)
    if detected:
        logger.info("Auto-detected %d existing transcripts", detected)
        save_csv(csv_path, episodes)

    if enable_diarization:
        diarized_detected = detect_existing_diarized_transcripts(episodes)
        if diarized_detected:
            logger.info(
                "Auto-detected %d existing diarized transcripts", diarized_detected
            )
            save_csv(csv_path, episodes)

    pipeline = None
    if enable_diarization:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set — diarization requires a HuggingFace token. "
                "Skipping diarization."
            )
            enable_diarization = False
        else:
            device = diarization_device or get_device()
            pipeline = load_pipeline(hf_token, device)

    if skip_transcription:
        if not enable_diarization:
            logger.warning("--skip-transcription requires --diarize. Nothing to do.")
            return

        diarize_pending = get_pending_diarization(episodes)
        diarize_pending = [
            ep for ep in diarize_pending if (audio_dir / ep["filename"]).exists()
        ]

        if not diarize_pending:
            logger.info("No episodes pending diarization.")
            return

        diarize_batch = diarize_pending[:limit] if limit else diarize_pending

        if dry_run:
            logger.info("Dry run: %d episodes would be diarized", len(diarize_batch))
            for i, ep in enumerate(diarize_batch, 1):
                logger.info("  [%d] %s", i, ep["title"])
            return

        start_time = time.time()
        logger.info(
            "Diarizing %d episodes (skip-transcription mode)", len(diarize_batch)
        )
        diarized = 0
        failed = 0
        for i, ep in enumerate(diarize_batch, 1):
            prefix = f"[{i}/{len(diarize_batch)}] "
            try:
                _diarize_episode(
                    ep,
                    audio_dir,
                    pipeline,
                    duration_limit_minutes=duration_limit_minutes,
                )
                diarized += 1
            except Exception as exc:
                logger.error("%sDiarization failed: %s — %s", prefix, ep["title"], exc)
                ep["diarization_status"] = "failed"
                failed += 1
            save_csv(csv_path, episodes)

        elapsed = time.time() - start_time
        minutes, secs = divmod(int(elapsed), 60)
        logger.info(
            "Batch complete: %d diarized, %d failed in %dm%02ds",
            diarized,
            failed,
            minutes,
            secs,
        )
        return

    pending = get_pending(episodes)
    pending = [ep for ep in pending if (audio_dir / ep["filename"]).exists()]

    if not pending:
        logger.info("No episodes pending transcription.")
        return

    batch = pending[:limit] if limit else pending

    if dry_run:
        logger.info("Dry run: %d episodes would be transcribed", len(batch))
        for i, ep in enumerate(batch, 1):
            logger.info("  [%d] %s", i, ep["title"])
        return

    logger.info(
        "Transcribing %d episodes (engine=%s, workers=%d, model=%s)",
        len(batch),
        engine,
        workers,
        model_size,
    )
    start_time = time.time()
    transcribed = 0
    failed = 0

    if workers > 1 and engine != "whisper.cpp":
        logger.warning(
            "Multi-worker mode only supported for whisper.cpp. "
            "Falling back to workers=1 for engine '%s'.",
            engine,
        )
        workers = 1

    if workers == 1:
        if engine == "whisper.cpp":
            n_threads = os.cpu_count() or 4
            model = load_whisper_cpp_model(model_size, n_threads)

        for i, ep in enumerate(batch, 1):
            audio_path = str(audio_dir / ep["filename"])
            prefix = f"[{i}/{len(batch)}] "
            try:
                if engine == "whisper.cpp":
                    result = transcribe_with_model(
                        model,
                        audio_path,
                        language,
                        initial_prompt,
                        log_prefix=prefix,
                        duration_limit_minutes=duration_limit_minutes,
                    )
                else:
                    result = transcribe(
                        audio_path,
                        model_size,
                        language,
                        engine,
                        initial_prompt=initial_prompt,
                        duration_limit_minutes=duration_limit_minutes,
                    )
                json_path, _ = save_output(result, audio_path)
                ep["transcription_status"] = "transcribed"
                ep["transcript_path"] = str(json_path)
                transcribed += 1

                if enable_diarization:
                    try:
                        _diarize_episode(
                            ep,
                            audio_dir,
                            pipeline,
                            duration_limit_minutes=duration_limit_minutes,
                        )
                    except Exception as exc:
                        logger.error(
                            "%sDiarization failed: %s — %s", prefix, ep["title"], exc
                        )
                        ep["diarization_status"] = "failed"

            except Exception as exc:
                logger.error("%sFailed: %s — %s", prefix, ep["title"], exc)
                ep["transcription_status"] = "failed"
                failed += 1
            save_csv(csv_path, episodes)
    else:
        n_threads = compute_threads_per_worker(workers)
        logger.info("Parallel mode: %d workers, %d threads each", workers, n_threads)
        logger.info(
            "Loading models in %d worker processes (this takes ~10s)...", workers
        )
        worker_counter = mp.Value("i", 0)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(worker_counter, model_size, n_threads),
        ) as executor:
            future_to_ep = {}
            for idx, ep in enumerate(batch, 1):
                audio_path = str(audio_dir / ep["filename"])
                prefix = f"[{idx}/{len(batch)}] "
                future = executor.submit(
                    _transcribe_episode,
                    audio_path,
                    language,
                    initial_prompt,
                    prefix,
                    duration_limit_minutes,
                )
                future_to_ep[future] = ep

            for i, future in enumerate(as_completed(future_to_ep), 1):
                ep = future_to_ep[future]
                try:
                    result = future.result()
                    ep["transcription_status"] = "transcribed"
                    ep["transcript_path"] = result["json_path"]
                    transcribed += 1
                    logger.info("[%d/%d] Transcribed: %s", i, len(batch), ep["title"])
                except Exception as exc:
                    ep["transcription_status"] = "failed"
                    failed += 1
                    logger.error(
                        "[%d/%d] Failed: %s — %s", i, len(batch), ep["title"], exc
                    )
                save_csv(csv_path, episodes)

        if enable_diarization:
            diarize_batch = [
                ep for ep in batch if ep.get("transcription_status") == "transcribed"
            ]
            logger.info(
                "Diarizing %d transcribed episodes sequentially...", len(diarize_batch)
            )
            for i, ep in enumerate(diarize_batch, 1):
                prefix = f"[D {i}/{len(diarize_batch)}] "
                try:
                    _diarize_episode(
                        ep,
                        audio_dir,
                        pipeline,
                        duration_limit_minutes=duration_limit_minutes,
                    )
                except Exception as exc:
                    logger.error(
                        "%sDiarization failed: %s — %s", prefix, ep["title"], exc
                    )
                    ep["diarization_status"] = "failed"
                save_csv(csv_path, episodes)

    elapsed = time.time() - start_time
    minutes, secs = divmod(int(elapsed), 60)
    logger.info(
        "Batch complete: %d transcribed, %d failed in %dm%02ds",
        transcribed,
        failed,
        minutes,
        secs,
    )

    if enable_diarization:
        diarized_count = sum(
            1 for e in batch if e.get("diarization_status") == "diarized"
        )
        diarize_failed = sum(
            1 for e in batch if e.get("diarization_status") == "failed"
        )
        logger.info(
            "Diarization: %d diarized, %d failed", diarized_count, diarize_failed
        )


def main():
    log_path = setup_logging(log_dir=LOGS_DIR, name="batch_transcribe")

    parser = argparse.ArgumentParser(description="Batch transcription pipeline")
    parser.add_argument(
        "--model", default="large", help="Whisper model size (default: large)"
    )
    parser.add_argument(
        "--engine",
        default="whisper.cpp",
        choices=ENGINES,
        help="Whisper engine (default: faster-whisper)",
    )
    parser.add_argument("--language", default="pt", help="Force language (default: pt)")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max episodes to transcribe"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transcribed without running",
    )
    parser.add_argument(
        "--no-prompt", action="store_true", help="Disable initial prompt"
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization after transcription",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip transcription, only diarize already-transcribed episodes (requires --diarize)",
    )
    parser.add_argument(
        "--diarization-device",
        default=None,
        help="Force diarization device: cpu, cuda, or mps (default: auto-detect)",
    )
    parser.add_argument(
        "--duration-limit",
        type=float,
        default=None,
        help="Limit transcription to the first N minutes of audio (e.g., 2 for 2 minutes)",
    )
    args = parser.parse_args()

    if args.workers >= 4:
        logger.warning(
            "Using %d workers may cause memory pressure on 32GB systems. Consider --workers 2 or 3.",
            args.workers,
        )

    logger.info("Log file: %s", log_path)

    prompt = None if args.no_prompt else INITIAL_PROMPT
    batch_transcribe(
        csv_path=EPISODES_CSV,
        audio_dir=AUDIO_DIR,
        model_size=args.model,
        engine=args.engine,
        language=args.language,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
        initial_prompt=prompt,
        enable_diarization=args.diarize,
        skip_transcription=args.skip_transcription,
        diarization_device=args.diarization_device,
        duration_limit_minutes=args.duration_limit,
    )


if __name__ == "__main__":
    main()
