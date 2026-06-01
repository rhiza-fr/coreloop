"""Logging configuration for coreloop."""

import logging

from rich.logging import RichHandler

_LOGGER_NAME = "coreloop"


def setup_logging(level: str | int = logging.WARNING) -> None:
    """Configure the coreloop logger with a RichHandler."""
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
