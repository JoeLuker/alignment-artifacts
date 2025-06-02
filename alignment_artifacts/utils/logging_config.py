"""Logging configuration for alignment artifacts library."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_style: str = "simple") -> logging.Logger:
    """
    Set up logging for the alignment artifacts library.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_style: Format style ('simple', 'detailed')

    Returns:
        Configured logger
    """

    # Create logger
    logger = logging.getLogger("alignment_artifacts")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatters
    if format_style == "detailed":
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:  # simple
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f"alignment_artifacts.{name}")
    return logging.getLogger("alignment_artifacts")


# Set up default logger
_default_logger = None


def set_verbosity(level: str):
    """Set global verbosity level."""
    global _default_logger
    _default_logger = setup_logging(level)
    return _default_logger


def get_default_logger() -> logging.Logger:
    """Get the default logger, creating it if necessary."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging("INFO")
    return _default_logger
