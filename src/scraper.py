"""Podcast RSS scraper and batch audio downloader.

Scrapes the NerdCast RSS feed, maintains a CSV catalog of all episodes,
and downloads MP3 files with resume support.

Usage:
    python src/scraper.py                  # sync feed + download all pending
    python src/scraper.py --sync-only      # update CSV catalog without downloading
    python src/scraper.py --limit 10       # download up to 10 pending episodes
    python src/scraper.py --feed-url URL   # use a custom RSS feed URL

The CSV at data/jovem-nerd-episodes.csv tracks each episode's title, URL,
publication date, duration, download status, filename, size, and format.
Downloads are resumable: re-running skips already-downloaded episodes and
picks up from the next pending one.
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import feedparser
import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

FEED_URL = "https://jn-feed.vercel.app/api/filter?podcast=nerdcast"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
EPISODES_CSV = DATA_DIR / "jovem-nerd-episodes.csv"
CSV_FIELDS = ["title", "url", "pub_date", "duration", "status", "filename", "size_mb", "format", "guid"]
USER_AGENT = "PodcastKnowledgeExtractor/1.0"


def _sanitize_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    return safe.strip()[:120]


def fetch_episodes(feed_url: str) -> list[dict]:
    logger.info("Fetching feed: %s", feed_url)
    feed = feedparser.parse(feed_url, agent=USER_AGENT)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse feed: {feed.bozo_exception}")

    episodes = []
    for entry in feed.entries:
        enclosures = getattr(entry, "enclosures", [])
        audio_url = ""
        for enc in enclosures:
            if enc.get("type", "").startswith("audio/") or enc.get("href", "").endswith(".mp3"):
                audio_url = enc.get("href", "")
                break
        if not audio_url and enclosures:
            audio_url = enclosures[0].get("href", "")
        if not audio_url:
            continue

        pub_parsed = getattr(entry, "published_parsed", None)
        pub_date = time.strftime("%Y-%m-%d", pub_parsed) if pub_parsed else ""

        duration_secs = 0
        itunes_duration = getattr(entry, "itunes_duration", None)
        if itunes_duration:
            try:
                duration_secs = int(itunes_duration)
            except ValueError:
                parts = itunes_duration.split(":")
                try:
                    if len(parts) == 3:
                        duration_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    elif len(parts) == 2:
                        duration_secs = int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    pass

        minutes, seconds = divmod(duration_secs, 60)

        episodes.append({
            "title": getattr(entry, "title", "Untitled").strip(),
            "url": audio_url,
            "pub_date": pub_date,
            "duration": f"{minutes}m{seconds:02d}s",
            "status": "not_downloaded",
            "filename": "",
            "size_mb": "",
            "format": "",
            "guid": getattr(entry, "id", audio_url),
        })

    logger.info("Found %d episodes in feed", len(episodes))
    return episodes


def load_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(csv_path: Path, episodes: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_eps = sorted(episodes, key=lambda e: e.get("pub_date", ""), reverse=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_eps)


def sync_feed_to_csv(feed_url: str, csv_path: Path) -> list[dict]:
    feed_episodes = fetch_episodes(feed_url)
    existing = load_csv(csv_path)

    existing_by_guid = {ep["guid"]: ep for ep in existing}

    merged = []
    seen_guids = set()
    for ep in feed_episodes:
        guid = ep["guid"]
        seen_guids.add(guid)
        if guid in existing_by_guid:
            old = existing_by_guid[guid]
            old["title"] = ep["title"]
            old["url"] = ep["url"]
            old["pub_date"] = ep["pub_date"]
            old["duration"] = ep["duration"]
            merged.append(old)
        else:
            merged.append(ep)

    for ep in existing:
        if ep["guid"] not in seen_guids:
            merged.append(ep)

    save_csv(csv_path, merged)

    total = len(merged)
    new = total - len(existing_by_guid)
    downloaded = sum(1 for e in merged if e["status"] == "downloaded")
    pending = sum(1 for e in merged if e["status"] in ("not_downloaded", "failed"))
    logger.info("CSV synced: %d total, %d new, %d downloaded, %d pending", total, new, downloaded, pending)

    return sorted(merged, key=lambda e: e.get("pub_date", ""), reverse=True)


def download_episode(episode: dict, audio_dir: Path, client: httpx.Client) -> dict:
    if episode["status"] == "downloaded" and episode.get("filename"):
        filepath = audio_dir / episode["filename"]
        if filepath.exists():
            logger.debug("Already downloaded: %s", episode["title"])
            return episode

    filename = _sanitize_filename(episode["title"]) + ".mp3"
    filepath = audio_dir / filename

    if filepath.exists():
        episode["status"] = "downloaded"
        episode["filename"] = filename
        logger.info("File already on disk: %s", filename)
        return episode

    part_path = filepath.with_suffix(".mp3.part")
    logger.info("Downloading: %s", episode["title"])

    try:
        with client.stream("GET", episode["url"], follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None

            with open(part_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename[:60], leave=True
            ) as pbar:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

        part_path.rename(filepath)
        episode["status"] = "downloaded"
        episode["filename"] = filename
        size_mb = filepath.stat().st_size / (1024 * 1024)
        episode["size_mb"] = f"{size_mb:.1f}"
        episode["format"] = filepath.suffix.lstrip(".")
        logger.info("Saved: %s (%.1f MB)", filename, size_mb)

    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Failed to download %s: %s", episode["title"], exc)
        episode["status"] = "failed"
        if part_path.exists():
            part_path.unlink()

    return episode


def download_pending(
    episodes: list[dict], audio_dir: Path, csv_path: Path, limit: int | None = None
) -> None:
    pending = [ep for ep in episodes if ep["status"] in ("not_downloaded", "failed")]

    if not pending:
        logger.info("All episodes already downloaded.")
        return

    batch = pending[:limit] if limit else pending
    logger.info("Downloading %d of %d pending episodes", len(batch), len(pending))

    audio_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        follow_redirects=True,
    ) as client:
        for i, ep in enumerate(batch, 1):
            logger.info("[%d/%d] %s", i, len(batch), ep["title"])
            download_episode(ep, audio_dir, client)
            save_csv(csv_path, episodes)

    downloaded = sum(1 for e in batch if e["status"] == "downloaded")
    failed = sum(1 for e in batch if e["status"] == "failed")
    logger.info("Batch complete: %d downloaded, %d failed", downloaded, failed)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Podcast episode downloader with CSV tracking")
    parser.add_argument("--feed-url", default=FEED_URL, help="RSS feed URL")
    parser.add_argument("--sync-only", action="store_true", help="Sync feed to CSV without downloading")
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to download in this run")
    args = parser.parse_args()

    episodes = sync_feed_to_csv(args.feed_url, EPISODES_CSV)

    if args.sync_only:
        logger.info("Sync complete. CSV at %s", EPISODES_CSV)
        return

    download_pending(episodes, AUDIO_DIR, EPISODES_CSV, args.limit)
    logger.info("Done!")


if __name__ == "__main__":
    main()
