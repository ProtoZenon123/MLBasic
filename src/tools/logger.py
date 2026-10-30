"""Shared logger: progress in the terminal, results in a text file."""

import sys
from pathlib import Path

from loguru import logger

logger.level("RESULT", no=25)


def configure_logger(result_file: Path) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {message}",
        filter=lambda record: record["level"].name != "RESULT",
    )
    logger.add(
        result_file,
        level="RESULT",
        format="{message}",
        filter=lambda record: record["level"].name == "RESULT",
        mode="w",
        encoding="utf-8",
    )
