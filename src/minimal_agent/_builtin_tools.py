"""Backward-compat shim — tools now live in ``minimal_agent.tools``."""

from .tools import make_tools
from .tools._shared import _resolve_safe, _resolve_safe_strict, _fmt_size
from .tools.edit import _character_line, _find_occurrence_near_line, _MAX_EDIT_BYTES

__all__ = [
    "make_tools",
    "_resolve_safe",
    "_resolve_safe_strict",
    "_fmt_size",
    "_character_line",
    "_find_occurrence_near_line",
    "_MAX_EDIT_BYTES",
]
