"""Unit tests for the transcription module."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import src.transcribe as transcribe_mod
from src.transcribe import (
    INITIAL_PROMPT,
    _format_timestamp,
    _get_audio_duration,
    _get_openai_whisper_device,
    _transcribe_faster_whisper,
    _transcribe_whisper_cpp,
    save_output,
    transcribe,
)


class TestFormatTimestamp(unittest.TestCase):

    def test_format_timestamp_zero(self):
        self.assertEqual(_format_timestamp(0.0), "00:00")

    def test_format_timestamp_seconds_only(self):
        self.assertEqual(_format_timestamp(45.0), "00:45")

    def test_format_timestamp_minutes_and_seconds(self):
        self.assertEqual(_format_timestamp(125.0), "02:05")

    def test_format_timestamp_just_under_one_hour(self):
        self.assertEqual(_format_timestamp(3599.0), "59:59")

    def test_format_timestamp_exactly_one_hour(self):
        self.assertEqual(_format_timestamp(3600.0), "1:00:00")

    def test_format_timestamp_large_value(self):
        self.assertEqual(_format_timestamp(7384.0), "2:03:04")

    def test_format_timestamp_truncates_fractional(self):
        self.assertEqual(_format_timestamp(65.9), "01:05")


class TestTranscribeRouter(unittest.TestCase):

    DUMMY_RESULT = {"text": "", "segments": [], "language": "pt"}

    def test_dispatches_openai_whisper(self):
        with patch("src.transcribe._transcribe_openai_whisper", return_value=self.DUMMY_RESULT) as mock_fn:
            transcribe("audio.mp3", engine="openai-whisper")
            mock_fn.assert_called_once_with("audio.mp3", "base", None, initial_prompt=INITIAL_PROMPT)

    def test_dispatches_whisper_cpp(self):
        with patch("src.transcribe._transcribe_whisper_cpp", return_value=self.DUMMY_RESULT) as mock_fn:
            transcribe("audio.mp3", engine="whisper.cpp")
            mock_fn.assert_called_once_with("audio.mp3", "base", None, initial_prompt=INITIAL_PROMPT)

    def test_dispatches_faster_whisper(self):
        with patch("src.transcribe._transcribe_faster_whisper", return_value=self.DUMMY_RESULT) as mock_fn:
            transcribe("audio.mp3", engine="faster-whisper")
            mock_fn.assert_called_once_with("audio.mp3", "base", None, "float16", initial_prompt=INITIAL_PROMPT)

    def test_default_engine_is_faster_whisper(self):
        with patch("src.transcribe._transcribe_faster_whisper", return_value=self.DUMMY_RESULT) as mock_fn:
            transcribe("audio.mp3")
            mock_fn.assert_called_once()


class TestSaveOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_result = {
            "text": "  Hello world  ",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello"},
                {"start": 5.0, "end": 10.0, "text": "world"},
            ],
            "language": "pt",
        }

    def test_save_output_creates_files(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, json_path = save_output(self.sample_result, audio_path)
                self.assertTrue(txt_path.exists())
                self.assertTrue(json_path.exists())
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_txt_content_stripped(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, _ = save_output(self.sample_result, audio_path)
                content = txt_path.read_text(encoding="utf-8")
                self.assertEqual(content, "Hello world")
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_json_content(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                _, json_path = save_output(self.sample_result, audio_path)
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(len(data), 2)
                self.assertAlmostEqual(data[0]["start"], 0.0, places=5)
                self.assertAlmostEqual(data[0]["end"], 5.0, places=5)
                self.assertEqual(data[0]["text"], "Hello")
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_json_excludes_extra_keys(self):
        result_with_extras = {
            "text": "Hello",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello", "speaker": "SPEAKER_00", "confidence": 0.95}],
            "language": "pt",
        }
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                _, json_path = save_output(result_with_extras, audio_path)
                data = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(set(data[0].keys()), {"start", "end", "text"})
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_stem_extraction(self):
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode-42.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, json_path = save_output(self.sample_result, audio_path)
                self.assertEqual(txt_path.name, "episode-42.txt")
                self.assertEqual(json_path.name, "episode-42.json")
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir

    def test_save_output_utf8_special_chars(self):
        result = {
            "text": "Olá, açafrão e coração",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Olá, açafrão e coração"}],
            "language": "pt",
        }
        with TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "episode.mp3")
            original_dir = transcribe_mod.TRANSCRIPTS_DIR
            try:
                transcribe_mod.TRANSCRIPTS_DIR = Path(tmpdir) / "transcripts"
                txt_path, json_path = save_output(result, audio_path)
                self.assertIn("açafrão", txt_path.read_text(encoding="utf-8"))
                json_content = json_path.read_text(encoding="utf-8")
                self.assertIn("açafrão", json_content)
                self.assertNotIn("\\u", json_content)
            finally:
                transcribe_mod.TRANSCRIPTS_DIR = original_dir


class TestGetOpenaiWhisperDevice(unittest.TestCase):

    def test_device_mps_available(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertEqual(_get_openai_whisper_device(), "mps")

    def test_device_cuda_fallback(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertEqual(_get_openai_whisper_device(), "cuda")

    def test_device_cpu_fallback(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertEqual(_get_openai_whisper_device(), "cpu")


class TestGetAudioDuration(unittest.TestCase):

    def test_parses_stdout(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="123.456\n")
            result = _get_audio_duration("/path/to/audio.mp3")
        self.assertAlmostEqual(result, 123.456, places=3)

    def test_correct_ffprobe_command(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="60.0\n")
            _get_audio_duration("/path/to/audio.mp3")
            args = mock_run.call_args[0][0]
            self.assertEqual(args[0], "ffprobe")
            self.assertIn("/path/to/audio.mp3", args)
            self.assertIn("-show_entries", args)


class TestTranscribeFasterWhisper(unittest.TestCase):

    def _run_with_segments(self, mock_segments, info_language="pt"):
        mock_fw = MagicMock()
        mock_model = MagicMock()
        mock_info = MagicMock(language=info_language)
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        mock_fw.WhisperModel.return_value = mock_model
        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            return _transcribe_faster_whisper("audio.mp3", "base", None, "float16")

    def test_returns_expected_structure(self):
        result = self._run_with_segments([MagicMock(start=0.0, end=5.0, text="Hello")])
        self.assertIn("text", result)
        self.assertIn("segments", result)
        self.assertIn("language", result)
        self.assertEqual(result["language"], "pt")

    def test_segments_normalized(self):
        mock_seg = MagicMock(start=1.5, end=3.5, text="test")
        result = self._run_with_segments([mock_seg])
        seg = result["segments"][0]
        self.assertEqual(set(seg.keys()), {"start", "end", "text"})
        self.assertAlmostEqual(seg["start"], 1.5, places=5)
        self.assertAlmostEqual(seg["end"], 3.5, places=5)
        self.assertEqual(seg["text"], "test")

    def test_text_is_joined_stripped_segments(self):
        segs = [MagicMock(start=0.0, end=5.0, text=" Hello "), MagicMock(start=5.0, end=10.0, text=" world ")]
        result = self._run_with_segments(segs)
        self.assertEqual(result["text"], "Hello world")


class TestTranscribeWhisperCpp(unittest.TestCase):

    def _run_with_segments(self, mock_segments, model_size="base", language=None):
        mock_pwc_model_mod = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe.return_value = mock_segments
        mock_pwc_model_mod.Model.return_value = mock_model_instance
        mock_pwc = MagicMock()
        mock_pwc.model = mock_pwc_model_mod
        with patch.dict("sys.modules", {"pywhispercpp": mock_pwc, "pywhispercpp.model": mock_pwc_model_mod}):
            with patch("src.transcribe._get_audio_duration", return_value=120.0):
                result = _transcribe_whisper_cpp("audio.mp3", model_size, language)
        return result, mock_pwc_model_mod

    def test_centisecond_conversion(self):
        mock_seg = MagicMock(t0=500, t1=1000, text=" Hello ")
        result, _ = self._run_with_segments([mock_seg])
        self.assertAlmostEqual(result["segments"][0]["start"], 5.0, places=1)
        self.assertAlmostEqual(result["segments"][0]["end"], 10.0, places=1)

    def test_model_map_large(self):
        mock_seg = MagicMock(t0=0, t1=100, text="test")
        _, mock_mod = self._run_with_segments([mock_seg], model_size="large")
        model_arg = mock_mod.Model.call_args[0][0]
        self.assertEqual(model_arg, "large-v3")

    def test_model_map_passthrough(self):
        mock_seg = MagicMock(t0=0, t1=100, text="test")
        _, mock_mod = self._run_with_segments([mock_seg], model_size="base")
        model_arg = mock_mod.Model.call_args[0][0]
        self.assertEqual(model_arg, "base")

    def test_language_fallback_auto(self):
        mock_seg = MagicMock(t0=0, t1=100, text="test")
        result, _ = self._run_with_segments([mock_seg], language=None)
        self.assertEqual(result["language"], "auto")


if __name__ == "__main__":
    unittest.main()
