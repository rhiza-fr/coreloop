from __future__ import annotations

import asyncio
from pathlib import Path

from .._tool import ToolInfo
from ._shared import _resolve_safe_strict, _make_tool_info

def make_search_tool(root: str, *, max_chars: int = 20_000, search_timeout: float = 30.0) -> ToolInfo:
    root_path = Path(root).resolve()

    async def search(
        pattern: str,
        path: str = ".",
        file_type: str | None = None,
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
        file_type :
            Restrict search to files of this type (e.g. ``py``, ``js``, ``ts``).
        after_context :
            Number of lines to show after each match.
        files_with_matches :
            If true, only print paths of files that contain a match.
        """
        try:
            safe = _resolve_safe_strict(path, root_path)
        except ValueError as exc:
            return f"Error: {exc}"

        if after_context is not None and after_context < 0:
            return "Error: after_context must be >= 0"

        cmd = ["rg", "-e", pattern]
        if file_type:
            cmd += ["--type", file_type]
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
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=search_timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return "Error: search timed out after 30 seconds"
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
        if len(output) > max_chars:
            output = output[:max_chars]
            last_newline = output.rfind("\n")
            if last_newline > 0:
                output = output[: last_newline + 1]
            total_chars = len(stdout.decode("utf-8", errors="replace"))
            output += f"\n... (truncated — showing {max_chars} of {total_chars} characters)"
        return output

    return _make_tool_info(search)
