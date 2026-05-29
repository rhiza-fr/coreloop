import os
from pathlib import Path

from ..tool import ToolInfo
from .edit import make_edit_tool
from .ls import make_ls_tool
from .read import make_read_tool
from .search import make_search_tool


def make_tools(
    allowed_root: str | None = None,
    *,
    read_max_lines: int = 100,
    search_max_chars: int = 20_000,
    search_timeout: float = 30.0,
) -> list[ToolInfo]:
    """Build the built-in file tools scoped to *allowed_root*.

    If *allowed_root* is ``None`` the current working directory is used.
    """
    root = str(Path(allowed_root or os.getcwd()).resolve(strict=True))
    return [
        make_read_tool(root, max_lines=read_max_lines),
        make_ls_tool(root),
        make_edit_tool(root),
        make_search_tool(root, max_chars=search_max_chars, search_timeout=search_timeout),
    ]
