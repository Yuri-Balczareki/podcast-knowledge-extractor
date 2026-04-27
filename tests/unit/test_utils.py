"""Tests for src/utils.py logging utility."""

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils import setup_logging


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def tearDown(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_console_only_returns_none(self):
        result = setup_logging()
        self.assertIsNone(result)

    def test_console_only_adds_stream_handler(self):
        setup_logging()
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertIsInstance(root.handlers[0], logging.StreamHandler)

    def test_console_only_sets_info_level(self):
        setup_logging()
        root = logging.getLogger()
        self.assertEqual(root.level, logging.INFO)

    def test_file_logging_creates_log_file(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_path = setup_logging(log_dir=log_dir, name="test")
            self.assertIsNotNone(log_path)
            self.assertTrue(log_path.exists())
            self.assertTrue(log_path.name.startswith("test_"))
            self.assertTrue(log_path.suffix == ".log")

    def test_file_logging_adds_two_handlers(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir, name="test")
            root = logging.getLogger()
            self.assertEqual(len(root.handlers), 2)
            handler_types = {type(h) for h in root.handlers}
            self.assertIn(logging.StreamHandler, handler_types)
            self.assertIn(logging.FileHandler, handler_types)

    def test_file_logging_creates_directory(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            setup_logging(log_dir=log_dir, name="test")
            self.assertTrue(log_dir.exists())

    def test_log_format_includes_timestamp_and_level(self):
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_path = setup_logging(log_dir=log_dir, name="fmt")
            test_logger = logging.getLogger("test_format")
            test_logger.info("hello world")
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("INFO", content)
            self.assertIn("hello world", content)


if __name__ == "__main__":
    unittest.main()
