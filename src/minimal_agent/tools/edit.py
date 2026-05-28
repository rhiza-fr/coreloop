from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .._tool import ToolInfo
from ._shared import _resolve_safe_strict, _fmt_size, _make_tool_info

_MAX_EDIT_BYTES = 10 * 1024 * 1024  # 10 MB


def _character_line(content: str, pos: int) -> int:
    """Return the 1-based line number for character position *pos* in *content*."""
    return content[:pos].count("\n") + 1


def _find_occurrence_near_line(content: str, old_text: str, target_line: int) -> int:
    """Find the first occurrence of *old_text* whose 1-based line number
    matches *target_line*.  Returns the character index or ``-1``."""
    idx = -1
    while True:
        idx = content.find(old_text, idx + 1)
        if idx < 0:
            break
        if _character_line(content, idx) == target_line:
            return idx
    return -1


def make_edit_tool(root: str) -> ToolInfo:
    root_path = Path(root).resolve()

    async def edit(
        path: str,
        old_text: str,
        new_text: str,
        line_hint: int | None = None,
    ) -> str:
        """Replace *old_text* with *new_text* in a file (single replacement only).

        If *old_text* appears multiple times, ``line_hint`` is required to
        disambiguate.  Pass the 1-based line number where the text to
        replace appears.

        Parameters
        ----------
        path :
            Relative or absolute path to the file.
        old_text :
            Exact text to search for (must appear exactly once, unless
            ``line_hint`` is given).
        new_text :
            Replacement text.
        line_hint :
            Optional 1-based line number that *old_text* appears on.
            Required when *old_text* appears more than once.
        """
        try:
            safe = _resolve_safe_strict(path, root_path)
        except ValueError as exc:
            return f"Error: {exc}"

        if not old_text:
            return "Error: old_text must be a non-empty string"

        try:
            file_size = os.path.getsize(safe)
        except OSError as exc:
            return f"Error: cannot stat {path!r}: {exc}"

        if file_size > _MAX_EDIT_BYTES:
            return (
                f"Error: {path!r} is too large to edit "
                f"({_fmt_size(file_size)} > {_fmt_size(_MAX_EDIT_BYTES)})"
            )

        try:
            with open(safe, encoding="utf-8", errors="strict") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: {path!r} contains non-UTF-8 bytes; cannot edit"
        except OSError as exc:
            return f"Error: cannot read {path!r}: {exc}"

        if old_text not in content:
            return f"Error: old_text not found in {path!r}"

        count = content.count(old_text)

        if count == 1:
            idx = content.find(old_text)
        elif line_hint is not None:
            idx = _find_occurrence_near_line(content, old_text, line_hint)
            if idx < 0:
                return (
                    f"Error: old_text {old_text!r} appears {count} times in "
                    f"{path!r} but none are on line {line_hint}"
                )
        else:
            return (
                f"Error: old_text {old_text!r} appears {count} times in "
                f"{path!r}. Provide line_hint to specify which occurrence "
                f"to replace."
            )

        new_content = content[:idx] + new_text + content[idx + len(old_text):]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=os.path.dirname(safe), delete=False, encoding="utf-8", suffix=".tmp"
            ) as tmp:
                tmp.write(new_content)
                tmp_path = tmp.name
            os.replace(tmp_path, safe)
        except OSError as exc:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return f"Error: cannot write {path!r}: {exc}"

        return f"Replaced 1 occurrence in {path!r}"

    return _make_tool_info(edit)
