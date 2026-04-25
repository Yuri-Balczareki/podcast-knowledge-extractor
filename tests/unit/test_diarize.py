"""Unit tests for the speaker diarization merge algorithm and utilities."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.diarize import load_transcript, merge, save_output


class TestMerge(unittest.TestCase):

    def test_merge_single_speaker(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "World"},
        ]
        diarization = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
        result = merge(transcript, diarization)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_00")

    def test_merge_multiple_speakers(self):
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "First speaker"},
            {"start": 5.0, "end": 10.0, "text": "Second speaker"},
        ]
        diarization = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        result = merge(transcript, diarization)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")

    def test_merge_no_overlap(self):
        transcript = [{"start": 20.0, "end": 25.0, "text": "No overlap"}]
        diarization = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
        result = merge(transcript, diarization)
        self.assertEqual(result[0]["speaker"], "UNKNOWN")

    def test_merge_overlapping_diarization_max_wins(self):
        transcript = [{"start": 3.0, "end": 8.0, "text": "Overlap test"}]
        diarization = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        result = merge(transcript, diarization)
        self.assertEqual(result[0]["speaker"], "SPEAKER_01")

    def test_merge_preserves_original_fields(self):
        transcript = [{"start": 0.0, "end": 5.0, "text": "Keep me"}]
        diarization = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        result = merge(transcript, diarization)
        self.assertIn("start", result[0])
        self.assertIn("end", result[0])
        self.assertIn("text", result[0])
        self.assertIn("speaker", result[0])
        self.assertAlmostEqual(result[0]["start"], 0.0, places=5)
        self.assertAlmostEqual(result[0]["end"], 5.0, places=5)
        self.assertEqual(result[0]["text"], "Keep me")

    def test_merge_empty_transcript(self):
        result = merge([], [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}])
        self.assertEqual(result, [])

    def test_merge_empty_diarization(self):
        transcript = [{"start": 0.0, "end": 5.0, "text": "Alone"}]
        result = merge(transcript, [])
        self.assertEqual(result[0]["speaker"], "UNKNOWN")

    def test_merge_accumulated_overlap_across_fragments(self):
        transcript = [{"start": 0.0, "end": 10.0, "text": "Long segment"}]
        diarization = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        result = merge(transcript, diarization)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")


class TestLoadTranscript(unittest.TestCase):

    def test_load_transcript_valid_json(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = [{"start": 0.0, "end": 5.0, "text": "Hello"}]
            path.write_text(json.dumps(data), encoding="utf-8")
            result = load_transcript(path)
            self.assertEqual(result, data)

    def test_load_transcript_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_transcript("/nonexistent/path.json")

    def test_load_transcript_invalid_json(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not valid json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_transcript(path)


class TestSaveOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello world", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "Goodbye world", "speaker": "SPEAKER_01"},
        ]

    def test_save_output_creates_files(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            import src.diarize as diarize_mod

            original_dir = diarize_mod.TRANSCRIPTS_DIR
            try:
                diarize_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, json_path = save_output(self.segments, audio_path)
                self.assertTrue(txt_path.exists())
                self.assertTrue(json_path.exists())
            finally:
                diarize_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_json_format(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            import src.diarize as diarize_mod

            original_dir = diarize_mod.TRANSCRIPTS_DIR
            try:
                diarize_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                _, json_path = save_output(self.segments, audio_path)
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]["speaker"], "SPEAKER_00")
                self.assertEqual(data[1]["speaker"], "SPEAKER_01")
            finally:
                diarize_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_txt_format(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            import src.diarize as diarize_mod

            original_dir = diarize_mod.TRANSCRIPTS_DIR
            try:
                diarize_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, _ = save_output(self.segments, audio_path)
                lines = txt_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], "[SPEAKER_00] Hello world")
                self.assertEqual(lines[1], "[SPEAKER_01] Goodbye world")
            finally:
                diarize_mod.TRANSCRIPTS_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
