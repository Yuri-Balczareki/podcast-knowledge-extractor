"""Functional tests: run real models against a known audio clip and check quality thresholds."""

import json
import os
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
AUDIO_PATH = FIXTURES_DIR / "nerdcast_1025_clip.mp3"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth_1025.json"

_fixtures_available = AUDIO_PATH.exists() and GROUND_TRUTH_PATH.exists()
_hf_token = os.environ.get("HF_TOKEN", "")

skip_no_fixtures = pytest.mark.skipif(not _fixtures_available, reason="Functional test fixtures not found")
skip_no_hf_token = pytest.mark.skipif(not _hf_token, reason="HF_TOKEN not set, cannot load pyannote pipeline")


def _load_ground_truth() -> dict:
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def _build_pyannote_annotation(segments: list[dict]):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for seg in segments:
        annotation[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return annotation


def _compute_speaker_accuracy(merged_segments: list[dict], gt_diarization_segments: list[dict]) -> float:
    gt_speakers = sorted({s["speaker"] for s in gt_diarization_segments})
    hyp_speakers = sorted({s["speaker"] for s in merged_segments})

    overlap = defaultdict(lambda: defaultdict(float))
    for ms in merged_segments:
        for gs in gt_diarization_segments:
            o = max(0.0, min(ms["end"], gs["end"]) - max(ms["start"], gs["start"]))
            if o > 0:
                overlap[gs["speaker"]][ms["speaker"]] += o

    n = max(len(gt_speakers), len(hyp_speakers))
    cost = np.zeros((n, n))
    for i, gs in enumerate(gt_speakers):
        for j, hs in enumerate(hyp_speakers):
            cost[i][j] = -overlap[gs][hs]

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = -cost[row_ind, col_ind].sum()
    total = sum(s["end"] - s["start"] for s in gt_diarization_segments)
    return matched / total if total > 0 else 0.0


@pytest.mark.functional
@skip_no_fixtures
class TestTranscriptionQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.transcribe import transcribe

        cls.ground_truth = _load_ground_truth()
        cls.result = transcribe(
            str(AUDIO_PATH),
            model_size="large",
            language="pt",
            engine="whisper.cpp",
        )

    def test_transcription_returns_expected_structure(self):
        self.assertIn("text", self.result)
        self.assertIn("segments", self.result)
        self.assertIn("language", self.result)
        self.assertIsInstance(self.result["segments"], list)
        for seg in self.result["segments"]:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("text", seg)

    def test_transcription_wer_below_threshold(self):
        import jiwer

        reference = self.ground_truth["transcription"]["full_text"]
        hypothesis = self.result["text"]
        wer = jiwer.wer(reference, hypothesis)
        self.assertLess(wer, 0.50, f"WER {wer:.2%} exceeds 50% threshold")

    def test_transcription_detects_portuguese(self):
        self.assertEqual(self.result["language"], "pt")

    def test_transcription_segments_cover_audio_duration(self):
        expected_duration = self.ground_truth["duration_seconds"]
        last_end = max(seg["end"] for seg in self.result["segments"])
        self.assertAlmostEqual(last_end, expected_duration, delta=10.0)


@pytest.mark.functional
@skip_no_fixtures
@skip_no_hf_token
class TestDiarizationQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.diarize import diarize, load_pipeline

        cls.ground_truth = _load_ground_truth()
        num_speakers = len(cls.ground_truth["diarization"]["speakers"])
        pipeline = load_pipeline(_hf_token, "cpu")
        cls.diarization = diarize(
            str(AUDIO_PATH), pipeline,
            min_speakers=num_speakers, max_speakers=num_speakers,
        )

    def test_diarization_detects_expected_speaker_count(self):
        expected = len(self.ground_truth["diarization"]["speakers"])
        detected = len({seg["speaker"] for seg in self.diarization})
        self.assertAlmostEqual(detected, expected, delta=1)

    def test_diarization_der_below_threshold(self):
        from pyannote.metrics.diarization import DiarizationErrorRate

        reference = _build_pyannote_annotation(self.ground_truth["diarization"]["segments"])
        hypothesis = _build_pyannote_annotation(self.diarization)
        der_metric = DiarizationErrorRate()
        der = der_metric(reference, hypothesis)
        self.assertLess(der, 0.40, f"DER {der:.2%} exceeds 40% threshold")


@pytest.mark.functional
@skip_no_fixtures
@skip_no_hf_token
class TestMergeEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.diarize import diarize, load_pipeline, merge
        from src.transcribe import transcribe

        cls.ground_truth = _load_ground_truth()
        num_speakers = len(cls.ground_truth["diarization"]["speakers"])
        transcript_result = transcribe(
            str(AUDIO_PATH),
            model_size="large",
            language="pt",
            engine="whisper.cpp",
        )
        pipeline = load_pipeline(_hf_token, "cpu")
        diarization_segments = diarize(
            str(AUDIO_PATH), pipeline,
            min_speakers=num_speakers, max_speakers=num_speakers,
        )
        cls.merged = merge(transcript_result["segments"], diarization_segments)

    def test_merged_segments_have_speaker_labels(self):
        for seg in self.merged:
            self.assertIn("speaker", seg)

    def test_merged_speaker_accuracy_above_threshold(self):
        gt_segments = self.ground_truth["diarization"]["segments"]
        accuracy = _compute_speaker_accuracy(self.merged, gt_segments)
        self.assertGreater(accuracy, 0.60, f"Speaker accuracy {accuracy:.2%} below 60% threshold")
