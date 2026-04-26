"""Unit tests for the batch transcription pipeline."""

import csv
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.batch_transcribe import (
    compute_threads_per_worker,
    detect_existing_transcripts,
    get_pending,
)


def _make_episode(
    title="Test Episode",
    status="downloaded",
    filename="Test Episode.mp3",
    transcription_status="not_transcribed",
    transcript_path="",
    guid="test-guid-1",
):
    return {
        "title": title,
        "url": "https://example.com/test.mp3",
        "pub_date": "2026-01-01",
        "duration": "60m00s",
        "status": status,
        "filename": filename,
        "size_mb": "50.0",
        "format": "mp3",
        "guid": guid,
        "transcription_status": transcription_status,
        "transcript_path": transcript_path,
    }


class TestGetPending(unittest.TestCase):

    def test_includes_downloaded_not_transcribed(self):
        episodes = [_make_episode(status="downloaded", transcription_status="not_transcribed")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 1)

    def test_includes_failed_transcription_for_retry(self):
        episodes = [_make_episode(status="downloaded", transcription_status="failed")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 1)

    def test_excludes_already_transcribed(self):
        episodes = [_make_episode(status="downloaded", transcription_status="transcribed")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 0)

    def test_excludes_not_downloaded(self):
        episodes = [_make_episode(status="not_downloaded", transcription_status="not_transcribed")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 0)

    def test_excludes_failed_download(self):
        episodes = [_make_episode(status="failed", transcription_status="not_transcribed")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 0)

    def test_excludes_missing_filename(self):
        episodes = [_make_episode(status="downloaded", filename="")]
        result = get_pending(episodes)
        self.assertEqual(len(result), 0)

    def test_mixed_episodes(self):
        episodes = [
            _make_episode(title="Pending", status="downloaded", transcription_status="not_transcribed", guid="1"),
            _make_episode(title="Done", status="downloaded", transcription_status="transcribed", guid="2"),
            _make_episode(title="Not DL", status="not_downloaded", transcription_status="not_transcribed", guid="3"),
            _make_episode(title="Retry", status="downloaded", transcription_status="failed", guid="4"),
        ]
        result = get_pending(episodes)
        self.assertEqual(len(result), 2)
        titles = [ep["title"] for ep in result]
        self.assertIn("Pending", titles)
        self.assertIn("Retry", titles)


class TestDetectExistingTranscripts(unittest.TestCase):

    def test_detects_existing_json(self):
        with TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir)
            json_path = transcripts_dir / "Test Episode.json"
            json_path.write_text("[]", encoding="utf-8")

            episodes = [_make_episode(filename="Test Episode.mp3")]
            with mock.patch("src.batch_transcribe.TRANSCRIPTS_DIR", transcripts_dir):
                detected = detect_existing_transcripts(episodes)

            self.assertEqual(detected, 1)
            self.assertEqual(episodes[0]["transcription_status"], "transcribed")
            self.assertEqual(episodes[0]["transcript_path"], str(json_path))

    def test_skips_already_transcribed(self):
        with TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir)
            json_path = transcripts_dir / "Test Episode.json"
            json_path.write_text("[]", encoding="utf-8")

            episodes = [_make_episode(transcription_status="transcribed", transcript_path="/old/path.json")]
            with mock.patch("src.batch_transcribe.TRANSCRIPTS_DIR", transcripts_dir):
                detected = detect_existing_transcripts(episodes)

            self.assertEqual(detected, 0)
            self.assertEqual(episodes[0]["transcript_path"], "/old/path.json")

    def test_no_match_when_file_missing(self):
        with TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir)
            episodes = [_make_episode()]
            with mock.patch("src.batch_transcribe.TRANSCRIPTS_DIR", transcripts_dir):
                detected = detect_existing_transcripts(episodes)

            self.assertEqual(detected, 0)
            self.assertEqual(episodes[0]["transcription_status"], "not_transcribed")

    def test_skips_not_downloaded_episodes(self):
        with TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir)
            json_path = transcripts_dir / "Test Episode.json"
            json_path.write_text("[]", encoding="utf-8")

            episodes = [_make_episode(status="not_downloaded")]
            with mock.patch("src.batch_transcribe.TRANSCRIPTS_DIR", transcripts_dir):
                detected = detect_existing_transcripts(episodes)

            self.assertEqual(detected, 0)


class TestBackfillColumns(unittest.TestCase):

    def test_setdefault_adds_missing_keys(self):
        ep = {"title": "Old Episode", "status": "downloaded", "filename": "old.mp3", "guid": "g1"}
        ep.setdefault("transcription_status", "not_transcribed")
        ep.setdefault("transcript_path", "")
        self.assertEqual(ep["transcription_status"], "not_transcribed")
        self.assertEqual(ep["transcript_path"], "")

    def test_preserves_existing_values(self):
        ep = {
            "title": "Done Episode",
            "transcription_status": "transcribed",
            "transcript_path": "/data/transcripts/done.json",
        }
        ep.setdefault("transcription_status", "not_transcribed")
        ep.setdefault("transcript_path", "")
        self.assertEqual(ep["transcription_status"], "transcribed")
        self.assertEqual(ep["transcript_path"], "/data/transcripts/done.json")


class TestThreadAllocation(unittest.TestCase):

    def test_single_worker_gets_all_threads(self):
        with mock.patch("os.cpu_count", return_value=10):
            self.assertEqual(compute_threads_per_worker(1), 10)

    def test_two_workers_split_threads(self):
        with mock.patch("os.cpu_count", return_value=10):
            self.assertEqual(compute_threads_per_worker(2), 5)

    def test_three_workers_split_threads(self):
        with mock.patch("os.cpu_count", return_value=10):
            self.assertEqual(compute_threads_per_worker(3), 3)

    def test_more_workers_than_cores_gives_one(self):
        with mock.patch("os.cpu_count", return_value=4):
            self.assertEqual(compute_threads_per_worker(8), 1)

    def test_cpu_count_none_defaults_to_four(self):
        with mock.patch("os.cpu_count", return_value=None):
            self.assertEqual(compute_threads_per_worker(2), 2)


class TestBatchTranscribeDryRun(unittest.TestCase):

    def test_dry_run_does_not_modify_csv(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "episodes.csv"
            audio_dir = tmpdir / "audio"
            audio_dir.mkdir()

            ep = _make_episode()
            (audio_dir / ep["filename"]).write_bytes(b"fake audio")

            from src.scraper import CSV_FIELDS

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(ep)

            original = csv_path.read_text(encoding="utf-8")

            from src.batch_transcribe import batch_transcribe

            with mock.patch("src.batch_transcribe.TRANSCRIPTS_DIR", tmpdir / "transcripts"):
                batch_transcribe(
                    csv_path=csv_path,
                    audio_dir=audio_dir,
                    dry_run=True,
                )

            self.assertEqual(csv_path.read_text(encoding="utf-8"), original)
