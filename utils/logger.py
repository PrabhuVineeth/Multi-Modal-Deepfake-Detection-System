"""
Logging configuration for the Deepfake Forensic Detection System.

Uses Loguru for structured, colored logging with file and console sinks.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure the global logger with console and optional file sinks.

    Args:
        log_dir: Directory for log files. If None, only console logging.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        rotation: Log file rotation size.
        retention: How long to keep old log files.
    """
    # Remove default handler
    logger.remove()

    # Console sink with color
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # File sink
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # General log
        logger.add(
            str(log_path / "forensics.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

        # Error-only log
        logger.add(
            str(log_path / "errors.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )

    logger.info(f"Logger initialized (level={level})")


def get_logger(name: str = "deepfake_forensics"):
    """
    Get a contextualized logger instance.

    Args:
        name: Logger context name.

    Returns:
        Loguru logger bound with the given name context.
    """
    return logger.bind(name=name)
