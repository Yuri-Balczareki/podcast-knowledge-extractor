"""Batch transcription pipeline for podcast episodes.

Reads the episode CSV, identifies downloaded-but-not-transcribed episodes,
and transcribes them using whisper.cpp with the large model. Supports
parallel workers and tracks progress in the CSV.

Usage:
    python src/batch_transcribe.py                    # transcribe all pending
    python src/batch_transcribe.py --limit 10         # transcribe up to 10 episodes
    python src/batch_transcribe.py --workers 2        # use 2 parallel workers
    python src/batch_transcribe.py --dry-run          # show what would be transcribed
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper import AUDIO_DIR, EPISODES_CSV, CSV_FIELDS, load_csv, save_csv
from src.transcribe import (
    INITIAL_PROMPT,
    TRANSCRIPTS_DIR,
    load_whisper_cpp_model,
    save_output,
    transcribe_with_model,
)

logger = logging.getLogger(__name__)

_worker_model = None
_worker_id = None


def _init_worker(counter, model_size: str, n_threads: int) -> None:
    global _worker_model, _worker_id
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    _worker_model = load_whisper_cpp_model(model_size, n_threads)


def _transcribe_episode(audio_path: str, language: str | None, initial_prompt: str | None, log_prefix: str) -> dict:
    full_prefix = f"[W{_worker_id}] {log_prefix}"
    result = transcribe_with_model(_worker_model, audio_path, language, initial_prompt, log_prefix=full_prefix)
    _, json_path = save_output(result, audio_path)
    return {"json_path": str(json_path)}


def detect_existing_transcripts(episodes: list[dict]) -> int:
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
    return [
        ep for ep in episodes
        if ep["status"] == "downloaded"
        and ep.get("transcription_status") in ("not_transcribed", "failed")
        and ep.get("filename")
    ]


def compute_threads_per_worker(workers: int) -> int:
    return max(1, (os.cpu_count() or 4) // workers)


def batch_transcribe(
    csv_path: Path,
    audio_dir: Path,
    model_size: str = "large",
    language: str | None = None,
    workers: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    initial_prompt: str | None = INITIAL_PROMPT,
) -> None:
    episodes = load_csv(csv_path)

    for ep in episodes:
        ep.setdefault("transcription_status", "not_transcribed")
        ep.setdefault("transcript_path", "")

    detected = detect_existing_transcripts(episodes)
    if detected:
        logger.info("Auto-detected %d existing transcripts", detected)
        save_csv(csv_path, episodes)

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

    logger.info("Transcribing %d episodes (workers=%d, model=%s)", len(batch), workers, model_size)
    start_time = time.time()
    transcribed = 0
    failed = 0

    if workers == 1:
        n_threads = os.cpu_count() or 4
        model = load_whisper_cpp_model(model_size, n_threads)
        for i, ep in enumerate(batch, 1):
            audio_path = str(audio_dir / ep["filename"])
            prefix = f"[{i}/{len(batch)}] "
            try:
                result = transcribe_with_model(model, audio_path, language, initial_prompt, log_prefix=prefix)
                _, json_path = save_output(result, audio_path)
                ep["transcription_status"] = "transcribed"
                ep["transcript_path"] = str(json_path)
                transcribed += 1
            except Exception as exc:
                logger.error("%sFailed: %s — %s", prefix, ep["title"], exc)
                ep["transcription_status"] = "failed"
                failed += 1
            save_csv(csv_path, episodes)
    else:
        n_threads = compute_threads_per_worker(workers)
        logger.info("Parallel mode: %d workers, %d threads each", workers, n_threads)
        logger.info("Loading models in %d worker processes (this takes ~10s)...", workers)
        worker_counter = mp.Value('i', 0)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(worker_counter, model_size, n_threads),
        ) as executor:
            future_to_ep = {}
            for idx, ep in enumerate(batch, 1):
                audio_path = str(audio_dir / ep["filename"])
                prefix = f"[{idx}/{len(batch)}] "
                future = executor.submit(_transcribe_episode, audio_path, language, initial_prompt, prefix)
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
                    logger.error("[%d/%d] Failed: %s — %s", i, len(batch), ep["title"], exc)
                save_csv(csv_path, episodes)

    elapsed = time.time() - start_time
    minutes, secs = divmod(int(elapsed), 60)
    logger.info("Batch complete: %d transcribed, %d failed in %dm%02ds", transcribed, failed, minutes, secs)


LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"


def _setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"batch_transcribe_{timestamp}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return log_path


def main():
    log_path = _setup_logging()

    parser = argparse.ArgumentParser(description="Batch transcription pipeline")
    parser.add_argument("--model", default="large", help="Whisper model size (default: large)")
    parser.add_argument("--language", default=None, help="Force language (e.g. 'pt')")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to transcribe")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be transcribed without running")
    parser.add_argument("--no-prompt", action="store_true", help="Disable initial prompt")
    args = parser.parse_args()

    if args.workers >= 4:
        logger.warning("Using %d workers may cause memory pressure on 32GB systems. Consider --workers 2 or 3.", args.workers)

    logger.info("Log file: %s", log_path)

    prompt = None if args.no_prompt else INITIAL_PROMPT
    batch_transcribe(
        csv_path=EPISODES_CSV,
        audio_dir=AUDIO_DIR,
        model_size=args.model,
        language=args.language,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
        initial_prompt=prompt,
    )


if __name__ == "__main__":
    main()
