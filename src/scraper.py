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
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import feedparser
import httpx
from tqdm import tqdm

from src.utils import setup_logging

logger = logging.getLogger(__name__)

FEED_URL = "https://jn-feed.vercel.app/api/filter?podcast=nerdcast"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
EPISODES_CSV = DATA_DIR / "jovem-nerd-episodes.csv"
CSV_FIELDS = [
    "title",
    "url",
    "pub_date",
    "duration",
    "status",
    "filename",
    "size_mb",
    "format",
    "guid",
    "speakers",
    "transcription_status",
    "transcript_path",
    "diarization_status",
    "diarized_transcript_path",
]
USER_AGENT = "PodcastKnowledgeExtractor/1.0"

_NON_SPEAKERS = {
    "NerdCast", "Nerdcast", "nerdcast", "NerdCash", "Nerd", "RPG",
}


class _BoldGroupExtractor(HTMLParser):
    """Extract consecutive <strong> text groups from <p> tags.

    Processes each <p> independently and keeps the first one with speaker names.
    """

    def __init__(self):
        super().__init__()
        self._in_strong = False
        self._in_p = False
        self._in_parens = False
        self.best_groups: list[str] = []
        self._current_groups: list[str] = []
        self._current: list[str] = []

    def _flush(self):
        if self._current:
            self._current_groups.append(" ".join(self._current))
            self._current = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._in_p = True
            self._current_groups = []
            self._current = []
            self._in_parens = False
        if tag == "strong" and self._in_p:
            self._in_strong = True

    def handle_endtag(self, tag):
        if tag == "strong":
            self._in_strong = False
        if tag == "p" and self._in_p:
            self._in_p = False
            self._flush()
            filtered = [n for n in self._current_groups if n and n not in _NON_SPEAKERS]
            if filtered and not self.best_groups:
                self.best_groups = filtered

    def handle_data(self, data):
        if not self._in_p:
            return
        if self._in_strong:
            if not self._in_parens:
                self._current.append(data.strip())
            return
        if self._in_parens:
            if ")" in data:
                self._in_parens = False
            return
        if "(" in data:
            self._flush()
            self._in_parens = True
            return
        if data.isspace():
            return
        text = data.strip()
        if text in ("e", ","):
            self._flush()
        elif text:
            self._flush()


_KNOWN_SPEAKERS = {
    "Alottoni", "Alexandre Ottoni", "Azaghal", "Tucano", "Gaveta", "Portuguesa",
    "Sr. K", "Sra. Jovem Nerd", "Carlos Voltor", "Katiucha Barcelos",
    "Marcelo Bassoli", "Pedro Pallotta", "Marcel Campos", "Eduardo Spohr",
    "Dubox", "Didi Braguinha", "Rex", "Mykel", "Gustavo Bufarah",
    "Altay de Souza", "Ana Arantes", "André Souza", "Atsjuão", "Cris Dias",
    "Guga Mafra", "Filipe Figueiredo", "Caio Gomes", "Julia Campos",
    "Patife", "SaninPlay", "Max Valarezo", "Natacha Litvinov",
}

_KNOWN_FIRST_NAMES = {n.split()[0] for n in _KNOWN_SPEAKERS}


def _extract_speakers_from_text(text: str) -> str:
    names = []
    for speaker in _KNOWN_SPEAKERS:
        if speaker in text:
            names.append(speaker)
    for speaker in list(names):
        first = speaker.split()[0]
        for other in names:
            if other != speaker and other.startswith(first) and len(other) > len(speaker):
                names.remove(speaker)
                break
    order = [(text.index(n), n) for n in names]
    order.sort()
    return "|".join(n for _, n in order)


def _extract_speakers(entry) -> str:
    """Extract speaker names from RSS entry's content:encoded HTML.

    Tries bold-tag extraction first; falls back to plain text pattern matching.
    Returns pipe-separated string, e.g. 'Alottoni|Azaghal|Sr. K'.
    """
    content = getattr(entry, "content", [])
    if not content:
        return ""
    html = content[0].get("value", "")
    if not html:
        return ""

    if "<strong>" in html:
        parser = _BoldGroupExtractor()
        parser.feed(html)
        if parser.best_groups:
            return "|".join(parser.best_groups)

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return _extract_speakers_from_text(text)


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
            if enc.get("type", "").startswith("audio/") or enc.get("href", "").endswith(
                ".mp3"
            ):
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
                        duration_secs = (
                            int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        )
                    elif len(parts) == 2:
                        duration_secs = int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    pass

        minutes, seconds = divmod(duration_secs, 60)

        episodes.append(
            {
                "title": getattr(entry, "title", "Untitled").strip(),
                "url": audio_url,
                "pub_date": pub_date,
                "duration": f"{minutes}m{seconds:02d}s",
                "status": "not_downloaded",
                "filename": "",
                "size_mb": "",
                "format": "",
                "guid": getattr(entry, "id", audio_url),
                "speakers": _extract_speakers(entry),
                "transcription_status": "not_transcribed",
                "transcript_path": "",
            }
        )

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

    for ep in existing:
        ep.setdefault("transcription_status", "not_transcribed")
        ep.setdefault("transcript_path", "")
        ep.setdefault("diarization_status", "not_diarized")
        ep.setdefault("diarized_transcript_path", "")
        ep.setdefault("speakers", "")

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
            old["speakers"] = ep["speakers"]
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
    logger.info(
        "CSV synced: %d total, %d new, %d downloaded, %d pending",
        total,
        new,
        downloaded,
        pending,
    )

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

            with (
                open(part_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=filename[:60],
                    leave=True,
                ) as pbar,
            ):
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
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Podcast episode downloader with CSV tracking"
    )
    parser.add_argument("--feed-url", default=FEED_URL, help="RSS feed URL")
    parser.add_argument(
        "--sync-only", action="store_true", help="Sync feed to CSV without downloading"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max episodes to download in this run"
    )
    args = parser.parse_args()

    episodes = sync_feed_to_csv(args.feed_url, EPISODES_CSV)

    if args.sync_only:
        logger.info("Sync complete. CSV at %s", EPISODES_CSV)
        return

    download_pending(episodes, AUDIO_DIR, EPISODES_CSV, args.limit)
    logger.info("Done!")


if __name__ == "__main__":
    main()
