"""read.py -- line-based file reader with offset/limit pagination."""

import itertools
from pathlib import Path

from ..tool_registry import ToolInfo
from ._shared import _resolve_safe_strict, _fmt_size, _make_tool_info


def make_read_tool(
    root: str, *, max_lines: int = 100, max_bytes: int = 10 * 1024 * 1024
) -> ToolInfo:
    """Build a read tool scoped to *root* with configurable line and byte limits."""
    root_path = Path(root).resolve()

    async def read(path: str, offset: int = 1, limit: int = max_lines) -> str:
        """Use this to read a file's contents. For large files, page through sections with offset and limit.

        Parameters
        ----------
        path :
            Relative or absolute path to the file.
        offset :
            Line number to start reading from (1-based).
        limit :
            Maximum number of lines to return (default 100).
        """
        try:
            safe = _resolve_safe_strict(path, root_path)
        except ValueError as exc:
            return f"Error: {exc}"

        _offset = max(0, int(offset) - 1)
        _limit = min(int(limit), max_lines)

        if _limit <= 0:
            return "Error: limit must be a positive integer"

        try:
            file_size = Path(safe).stat().st_size
        except OSError as exc:
            return f"Error: cannot stat {path!r}: {exc}"

        if file_size > max_bytes:
            return (
                f"Error: {path!r} is too large to read "
                f"({_fmt_size(file_size)} > {_fmt_size(max_bytes)})"
            )

        _stop = _offset + _limit

        try:
            with open(safe, encoding="utf-8", errors="replace") as f:
                lines = list(itertools.islice(f, _offset, _stop + 1))
        except OSError as exc:
            return f"Error: cannot read {path!r}: {exc}"

        if not lines:
            return ""

        truncated = len(lines) > _limit
        if truncated:
            lines = lines[:_limit]

        result = "".join(lines)
        if truncated:
            result += f"\n[Truncated: showing lines {_offset + 1}-{_offset + _limit}. Call again with offset={_offset + _limit + 1} to continue.]"
        return result

    return _make_tool_info(read)
