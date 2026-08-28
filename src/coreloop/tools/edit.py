"""edit.py -- atomic string-replace edit tool for text files."""

import asyncio
import os
import tempfile
from pathlib import Path

from ..tool_registry import ToolInfo
from ._shared import _fmt_size, _make_tool_info, _resolve_safe, _resolve_safe_strict


def make_edit_tool(root: str, *, max_bytes: int = 10 * 1024 * 1024) -> ToolInfo:
    """Build an edit tool scoped to *root* with an atomic write via a temp file."""
    root_path = Path(root).resolve()

    async def edit(
        path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """Use this to make a targeted edit to a file by replacing an exact string. To create a new file, pass old_text as an empty string.

        To create a new file, pass ``old_text=""`` - the file must not already
        exist.  Parent directories are created automatically.

        Parameters
        ----------
        path :
            Relative or absolute path to the file.
        old_text :
            Exact text to search for (must appear exactly once).
            Pass ``""`` to create a new file.
        new_text :
            Replacement text (or full content when creating a new file).
        """
        if not old_text:
            try:
                safe = _resolve_safe(path, root_path)
            except ValueError as exc:
                return f"Error: {exc}"
            if not Path(safe).exists():
                try:
                    Path(safe).parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        "w", dir=Path(safe).parent, delete=False, encoding="utf-8", suffix=".tmp"
                    ) as tmp:
                        tmp.write(new_text)
                        tmp_path = tmp.name
                    os.replace(tmp_path, safe)
                except OSError as exc:
                    return f"Error: cannot create {path!r}: {exc}"
                return f"Created {path!r}"
            # file exists and old_text is empty -- fall through to the empty-file check below
        else:
            try:
                safe = _resolve_safe_strict(path, root_path)
            except ValueError as exc:
                return f"Error: {exc}"

        try:
            file_size = Path(safe).stat().st_size
        except OSError as exc:
            return f"Error: cannot stat {path!r}: {exc}"

        if file_size > max_bytes:
            return (
                f"Error: {path!r} is too large to edit "
                f"({_fmt_size(file_size)} > {_fmt_size(max_bytes)})"
            )

        def _read_all() -> str:
            with open(safe, encoding="utf-8", errors="strict") as f:
                return f.read()

        try:
            content = await asyncio.to_thread(_read_all)
        except UnicodeDecodeError:
            return f"Error: {path!r} contains non-UTF-8 bytes; cannot edit"
        except OSError as exc:
            return f"Error: cannot read {path!r}: {exc}"

        if not old_text:
            if content:
                return (
                    f"Error: {path!r} already has content. "
                    "Pass old_text with the exact text to replace. "
                    "Empty old_text only works for creating new files."
                )
            idx = 0
        else:
            if old_text not in content:
                return f"Error: old_text not found in {path!r}"

            count = content.count(old_text)

            if count == 1:
                idx = content.find(old_text)
            else:
                return (
                    f"Error: old_text {old_text!r} appears {count} times in "
                    f"{path!r}. Use a longer, unique excerpt as old_text."
                )

        new_content = content[:idx] + new_text + content[idx + len(old_text) :]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=Path(safe).parent, delete=False, encoding="utf-8", suffix=".tmp"
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
