import os
from pathlib import Path

from ..registry import ToolInfo
from .edit import make_edit_tool
from .ls import make_ls_tool
from .read import make_read_tool
from .grep import make_grep_tool


def make_tools(
    allowed_root: str | Path | None = None,
    *,
    read_max_lines: int = 100,
    read_max_bytes: int = 10 * 1024 * 1024,
    ls_max_entries: int = 500,
    edit_max_bytes: int = 10 * 1024 * 1024,
    search_max_chars: int = 20_000,
    search_timeout: float = 30.0,
) -> list[ToolInfo]:
    """Build the built-in file tools scoped to *allowed_root*.

    If *allowed_root* is ``None`` the current working directory is used.
    """
    root = str(Path(allowed_root or os.getcwd()).resolve(strict=True))
    return [
        make_read_tool(root, max_lines=read_max_lines, max_bytes=read_max_bytes),
        make_ls_tool(root, max_entries=ls_max_entries),
        make_edit_tool(root, max_bytes=edit_max_bytes),
        make_grep_tool(root, max_chars=search_max_chars, search_timeout=search_timeout),
    ]
