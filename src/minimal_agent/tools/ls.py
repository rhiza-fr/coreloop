from __future__ import annotations

import os
from pathlib import Path

from .._tool import ToolInfo
from ._shared import _resolve_safe, _resolve_safe_strict, _fmt_size, _make_tool_info

_MAX_LS_ENTRIES = 500


def make_ls_tool(root: str) -> ToolInfo:
    root_path = Path(root).resolve()

    async def ls(path: str = ".") -> str:
        """List files and directories inside a directory.

        Parameters
        ----------
        path :
            Relative or absolute path to the directory.
        """
        try:
            safe = _resolve_safe_strict(path, root_path)
        except ValueError as exc:
            return f"Error: {exc}"

        safe_path = Path(safe)
        if not safe_path.is_dir():
            return f"Error: {path!r} is not a directory"

        try:
            entries = sorted(os.listdir(safe))
        except OSError as exc:
            return f"Error: cannot list {path!r}: {exc}"

        lines: list[str] = []
        for name in entries[:_MAX_LS_ENTRIES]:
            full = safe_path / name
            try:
                if full.is_symlink():
                    target = os.readlink(full)
                    abs_target = target if Path(target).is_absolute() else str(safe_path / target)
                    try:
                        _resolve_safe(abs_target, root_path)
                        lines.append(f"{name} -> {target}")
                    except ValueError:
                        lines.append(f"{name} -> <outside root>")
                elif full.is_dir():
                    lines.append(f"{name}/")
                else:
                    size = full.stat().st_size
                    lines.append(f"{name} ({_fmt_size(size)})")
            except OSError:
                lines.append(f"{name} (?)")

        remaining = len(entries) - _MAX_LS_ENTRIES
        if remaining > 0:
            lines.append(f"... ({remaining} more entries not shown)")

        return "\n".join(lines) if lines else "(empty directory)"

    return _make_tool_info(ls)
