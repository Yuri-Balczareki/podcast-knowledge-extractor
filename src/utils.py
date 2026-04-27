"""Shared utilities for the podcast knowledge extraction pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path


def setup_logging(log_dir: Path | None = None, name: str = "app") -> Path | None:
    """Configure root logger with console output and optional file logging."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        return log_path

    return None
