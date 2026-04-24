"""Podcast episode downloader — fetches RSS feed and downloads MP3s."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

FEED_URL = "https://jn-feed.vercel.app/api/filter?podcast=nerdcast"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"

NAMESPACES = {
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def fetch_feed(url: str) -> ET.Element:
    req = Request(url, headers={"User-Agent": "PodcastKnowledgeExtractor/1.0"})
    with urlopen(req, timeout=30) as resp:
        return ET.parse(resp).getroot()


def parse_episodes(root: ET.Element) -> list[dict]:
    episodes = []
    for item in root.iter("item"):
        enclosure = item.find("enclosure")
        if enclosure is None:
            continue

        duration_el = item.find("itunes:duration", NAMESPACES)
        duration_secs = int(duration_el.text) if duration_el is not None and duration_el.text else 0
        minutes, seconds = divmod(duration_secs, 60)

        episodes.append({
            "title": (item.findtext("title") or "Untitled").strip(),
            "date": (item.findtext("pubDate") or "").strip(),
            "duration": f"{minutes}m{seconds:02d}s",
            "duration_secs": duration_secs,
            "url": enclosure.get("url", ""),
            "guid": (item.findtext("guid") or "").strip(),
        })
    return episodes


def download_episode(episode: dict, output_dir: Path) -> Path:
    title = episode["title"]
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_name = safe_name.strip()[:120]
    filepath = output_dir / f"{safe_name}.mp3"

    if filepath.exists():
        print(f"Already exists: {filepath.name}")
        return filepath

    print(f"Downloading: {title}")
    print(f"  URL: {episode['url']}")

    req = Request(episode["url"], headers={"User-Agent": "PodcastKnowledgeExtractor/1.0"})
    with urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024

        with open(filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded / total * 100
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    mb_done = downloaded / (1024 * 1024)
                    print(f"\r  {mb_done:.1f} MB downloaded", end="", flush=True)

    print()
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Saved: {filepath.name} ({size_mb:.1f} MB)")
    return filepath


def main():
    print(f"Fetching feed: {FEED_URL}")
    root = fetch_feed(FEED_URL)
    episodes = parse_episodes(root)

    if not episodes:
        print("No episodes found in feed.")
        sys.exit(1)

    print(f"\nFound {len(episodes)} episodes.\n")
    print("Latest 10 episodes:")
    print("-" * 80)
    for i, ep in enumerate(episodes[:10]):
        print(f"  [{i}] {ep['title']}")
        print(f"      Date: {ep['date']}  |  Duration: {ep['duration']}")
    print("-" * 80)

    idx_input = input(f"\nWhich episode to download? [0-{len(episodes)-1}] (default: 0): ").strip()
    idx = int(idx_input) if idx_input else 0

    if idx < 0 or idx >= len(episodes):
        print(f"Invalid index: {idx}")
        sys.exit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    download_episode(episodes[idx], AUDIO_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
