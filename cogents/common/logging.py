"""
Simplified logging configuration for cogents with color support.

Usage:
    from cogents.common.logging import setup_logging, get_logger

    # Basic usage - automatically uses environment variables if available
    setup_logging()
    logger = get_logger(__name__)

    # Explicit configuration
    setup_logging(level="DEBUG", enable_colors=True)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    import colorlog
except ImportError:
    colorlog = None  # fallback if colorlog is not installed


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_file_level: Optional[str] = None,
    enable_colors: Optional[bool] = None,
    log_format: Optional[str] = None,
    custom_colors: Optional[Dict[str, str]] = None,
    third_party_level: Optional[str] = None,
    clear_existing: bool = True,
) -> None:
    """Initialize logging for cogents and third-party libs.

    Args:
        level: Log level (INFO, DEBUG, etc.)
        log_file: Path to log file
        log_file_level: Log level for file handler
        enable_colors: Whether to enable colored output
        log_format: Custom log format string
        custom_colors: Custom color mapping
        third_party_level: Log level for third-party libraries
        clear_existing: Whether to clear existing handlers
    """
    # Set defaults for any remaining None values
    level = level or "INFO"
    log_file_level = log_file_level or "DEBUG"
    enable_colors = enable_colors if enable_colors is not None else True
    third_party_level = third_party_level or "WARNING"

    root_logger = logging.getLogger()
    if clear_existing:
        root_logger.handlers.clear()

    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    fmt = log_format or "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    if enable_colors and colorlog:
        formatter = colorlog.ColoredFormatter(
            fmt=fmt,
            datefmt=datefmt,
            log_colors=custom_colors
            or {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            style="%",
        )
    else:
        formatter = logging.Formatter(fmt.replace("%(log_color)s", ""), datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_file_level.upper(), logging.DEBUG))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
