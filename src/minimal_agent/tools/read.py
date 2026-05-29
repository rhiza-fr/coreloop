import itertools
from pathlib import Path

from ..tool import ToolInfo
from ._shared import _resolve_safe_strict, _fmt_size, _make_tool_info, _MAX_READ_BYTES


def make_read_tool(root: str, *, max_lines: int = 100) -> ToolInfo:
    root_path = Path(root).resolve()

    async def read(path: str, offset: int = 1, limit: int = max_lines) -> str:
        """Read the contents of a text file.

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

        _offset = max(0, (int(offset) if offset is not None else 1) - 1)
        _limit = min(int(limit) if limit is not None else max_lines, max_lines)

        if _limit <= 0:
            return "Error: limit must be a positive integer"

        try:
            file_size = Path(safe).stat().st_size
        except OSError as exc:
            return f"Error: cannot stat {path!r}: {exc}"

        if file_size > _MAX_READ_BYTES:
            return (
                f"Error: {path!r} is too large to read "
                f"({_fmt_size(file_size)} > {_fmt_size(_MAX_READ_BYTES)})"
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
            result += f"\n[Truncated: showing lines {_offset + 1}-{_offset + _limit}. Use offset/limit to read more.]"
        return result

    return _make_tool_info(read)
