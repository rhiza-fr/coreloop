"""Built-in file-system tools: ``read``, ``ls``, ``edit``.

All tools enforce a safe root directory to prevent path traversal.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from pathlib import Path

from ._tool import ToolInfo

_MAX_EDIT_BYTES = 10 * 1024 * 1024  # 10 MB


def _resolve_safe(requested: str, root: str) -> str:
    """Resolve *requested* as an absolute path and verify it is under *root*.

    Raises ``ValueError`` (returned as a tool error message) if the path
    escapes the allowed root or does not exist (for reads/edits).
    """
    raw = Path(requested)
    if not raw.is_absolute():
        raw = Path(root) / raw

    try:
        # strict=False so we can resolve paths that don't exist yet (new files).
        # resolve() still follows symlinks for all existing components, so a
        # symlink inside the root that points outside it is caught by the
        # relative_to check below.
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"cannot resolve path {requested!r}: {exc}") from exc

    root_resolved = Path(root).resolve(strict=True)

    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"path traversal denied: {requested!r} resolves outside allowed root {root_resolved}"
        ) from None

    return str(resolved)


def _resolve_safe_strict(requested: str, root: str) -> str:
    """Like ``_resolve_safe`` but also requires the path to exist.

    The root/traversal check runs first (on the strict=False resolved path),
    then existence is verified so that traversal attempts always produce the
    'path traversal denied' error rather than a 'does not exist' message.
    """
    resolved = _resolve_safe(requested, root)
    if not os.path.exists(resolved):
        raise ValueError(f"path does not exist: {requested!r}")
    return resolved


# ── Tool factories ────────────────────────────────────────────


def make_tools(allowed_root: str | None = None) -> list[ToolInfo]:
    """Build the built-in file tools scoped to *allowed_root*.

    If *allowed_root* is ``None`` the current working directory is used.
    """
    root = str(Path(allowed_root or os.getcwd()).resolve(strict=True))

    # ── read ──────────────────────────────────────────────────

    async def read(path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read the contents of a text file.

        Parameters
        ----------
        path :
            Relative or absolute path to the file.
        offset :
            Line number to start reading from (1-based).
        limit :
            Maximum number of lines to return (``None`` = all lines).
        """
        try:
            safe = _resolve_safe_strict(path, root)
        except ValueError as exc:
            return f"Error: {exc}"

        # Coerce types — models sometimes pass strings instead of ints
        _offset = max(0, (int(offset) if offset is not None else 1) - 1)  # 0-based
        _limit: int | None = int(limit) if limit is not None else None
        _stop = _offset + _limit if _limit is not None else None

        try:
            with open(safe, encoding="utf-8", errors="replace") as f:
                lines = list(itertools.islice(f, _offset, _stop))
        except OSError as exc:
            return f"Error: cannot read {path!r}: {exc}"

        if not lines:
            return ""

        return "".join(lines)

    # ── ls ────────────────────────────────────────────────────

    async def ls(path: str = ".") -> str:
        """List files and directories inside a directory.

        Parameters
        ----------
        path :
            Relative or absolute path to the directory.
        """
        try:
            safe = _resolve_safe_strict(path, root)
        except ValueError as exc:
            return f"Error: {exc}"

        if not os.path.isdir(safe):
            return f"Error: {path!r} is not a directory"

        try:
            entries = sorted(os.listdir(safe))
        except OSError as exc:
            return f"Error: cannot list {path!r}: {exc}"

        lines: list[str] = []
        for name in entries:
            full = os.path.join(safe, name)
            try:
                if os.path.isdir(full):
                    lines.append(f"{name}/")
                elif os.path.islink(full):
                    target = os.readlink(full)
                    lines.append(f"{name} -> {target}")
                else:
                    size = os.path.getsize(full)
                    lines.append(f"{name} ({_fmt_size(size)})")
            except OSError:
                lines.append(f"{name} (?)")

        return "\n".join(lines) if lines else "(empty directory)"

    # ── edit ──────────────────────────────────────────────────

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
            safe = _resolve_safe_strict(path, root)
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
            with open(safe, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as exc:
            return f"Error: cannot read {path!r}: {exc}"

        if old_text not in content:
            return f"Error: old_text not found in {path!r}"

        count = content.count(old_text)

        # ── Resolve which occurrence to replace ──────────────
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

        # ── Perform the single replacement ───────────────────
        new_content = content[:idx] + new_text + content[idx + len(old_text):]

        try:
            with open(safe, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as exc:
            return f"Error: cannot write {path!r}: {exc}"

        return f"Replaced 1 occurrence in {path!r}"

    # ── search ────────────────────────────────────────────────

    async def search(
        pattern: str,
        path: str = ".",
        type: str | None = None,
        after_context: int | None = None,
        files_with_matches: bool = False,
    ) -> str:
        """Search for a regex pattern in files using ripgrep (rg).

        Parameters
        ----------
        pattern :
            Regular expression to search for.
        path :
            Directory or file to search in (relative to root).
        type :
            Restrict search to files of this type (e.g. ``py``, ``js``, ``ts``).
        after_context :
            Number of lines to show after each match.
        files_with_matches :
            If true, only print paths of files that contain a match.
        """
        try:
            safe = _resolve_safe_strict(path, root)
        except ValueError as exc:
            return f"Error: {exc}"

        cmd = ["rg", "-e", pattern]
        if type:
            cmd += ["--type", type]
        if after_context is not None:
            cmd += [f"--after-context={after_context}"]
        if files_with_matches:
            cmd.append("--files-with-matches")
        cmd.append(safe)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return "Error: 'rg' (ripgrep) not found on PATH"
        except OSError as exc:
            return f"Error: {exc}"

        if proc.returncode == 1:
            return "(no matches)"
        if proc.returncode not in (0, 1):
            err = stderr.decode("utf-8", errors="replace").strip()
            return f"Error: rg exited {proc.returncode}: {err}"

        output = stdout.decode("utf-8", errors="replace")
        _MAX_SEARCH_CHARS = 20_000
        if len(output) > _MAX_SEARCH_CHARS:
            output = output[:_MAX_SEARCH_CHARS]
            last_newline = output.rfind("\n")
            if last_newline > 0:
                output = output[: last_newline + 1]
            output += f"\n... (truncated — {len(stdout)} bytes total)"
        return output

    # ── Build ToolInfo list ───────────────────────────────────

    # We construct ToolInfo by hand instead of using @tool so that the
    # *root* is captured per-call (not at import time).

    def _info(fn, **overrides) -> ToolInfo:
        from ._tool import _infer_parameters
        name = overrides.pop("name", fn.__name__)
        desc = overrides.pop(
            "description",
            (fn.__doc__ or "").strip(),
        )
        params = _infer_parameters(fn)
        params.update(overrides.pop("parameters", {}))
        return ToolInfo(name=name, description=desc, parameters=params, fn=fn)

    return [
        _info(read),
        _info(ls),
        _info(edit),
        _info(search),
    ]


# ── Helpers ───────────────────────────────────────────────────

def _fmt_size(size: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _character_line(content: str, pos: int) -> int:
    """Return the 1-based line number for character position *pos* in *content*."""
    return content[:pos].count("\n") + 1


def _find_occurrence_near_line(
    content: str, old_text: str, target_line: int
) -> int:
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
