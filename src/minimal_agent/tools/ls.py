import os
from pathlib import Path

from ..tool_registry import ToolInfo
from ._shared import _resolve_safe, _resolve_safe_strict, _fmt_size, _make_tool_info


def make_ls_tool(root: str, *, max_entries: int = 500) -> ToolInfo:
    root_path = Path(root).resolve()

    async def ls(path: str = ".") -> str:
        """Use this to explore directory contents before reading or editing files.

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
        for name in entries[:max_entries]:
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

        remaining = len(entries) - max_entries
        if remaining > 0:
            lines.append(f"... ({remaining} more entries not shown)")

        return "\n".join(lines) if lines else "(empty directory)"

    return _make_tool_info(ls)
