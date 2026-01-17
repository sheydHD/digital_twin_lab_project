"""
Logging Setup Utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level
        log_file: Optional file path for logging output
    """
    # Create formatters
    console_format = "%(message)s"
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with Rich
    console_handler = RichHandler(
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(console_format))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)

    # Reduce verbosity of some libraries
    logging.getLogger("pymc").setLevel(logging.WARNING)
    logging.getLogger("pytensor").setLevel(logging.WARNING)
    logging.getLogger("arviz").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
