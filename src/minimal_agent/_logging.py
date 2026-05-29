"""Logging configuration for minimal-agent."""

import logging

from rich.logging import RichHandler

_LOGGER_NAME = "minimal_agent"


def setup_logging(level: str | int = logging.WARNING) -> None:
    """Configure the minimal_agent logger with a RichHandler."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return  # already configured

    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,  # avoid misinterpreting brackets/arrows in log messages
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
