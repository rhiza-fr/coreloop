"""bash.py — run shell commands via bash with safety guards.

Uses ``asyncio.create_subprocess_exec`` (matching ``grep.py``'s pattern)
with ``bash -c`` for portable command execution.

Safety features:

*   Dangerous command patterns are blocked with a clear error.
*   Working directory is validated within the allowed root.
*   Output is middle-truncated at a configurable char limit.
*   Timeout kills the entire process group (Unix) or the process (Windows).
"""

import asyncio
import os
import re
import signal
import sys
from pathlib import Path

from ..registry import ToolInfo
from ._shared import _make_tool_info, _resolve_safe_strict

DEFAULT_DANGEROUS_PATTERNS: list[str] = [
    r"\brm\s+-rf\s+/\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r":\(\)\s*\{",
    r"\bmkfs\.",
    r"\bmkswap\b",
    r"chmod\s+777\s+/",
    r">\s*/dev/sda",
]


def make_bash_tool(
    root: str,
    *,
    max_chars: int = 10_000,
    max_raw_bytes: int = 100 * 1024,
    default_timeout: int = 180,
    max_timeout: int = 300,
    dangerous_patterns: list[str] | None = None,
) -> ToolInfo:
    """Build a ``bash`` tool scoped to *root*.

    Parameters
    ----------
    root :
        Absolute path to the allowed root directory.  All commands and
        working-directory values are validated against this.
    max_chars :
        Output character cap before middle-truncation kicks in.
    """
    root_path = Path(root).resolve()
    _patterns = [re.compile(p) for p in (dangerous_patterns if dangerous_patterns is not None else DEFAULT_DANGEROUS_PATTERNS)]

    async def bash(
        command: str,
        timeout: int | None = default_timeout,
        workdir: str | None = None,
    ) -> str:
        """Execute a shell command via bash and return combined stdout+stderr.

        Safety features:

        - Dangerous command patterns are blocked with a clear error.
        - Working directory is validated to stay inside the allowed root.
        - Output is capped at 10,000 chars with middle truncation
          (preserves first 40% and last 40%, removes the middle).

        Parameters
        ----------
        command :
            Shell command to execute via ``bash -c``.
        timeout :
            Max seconds to wait for completion (default 180, max 300).
            The outer agent-level ``tool_timeout`` also applies as an
            additional safety net.
        workdir :
            Working directory for the command.  Defaults to the project
            root.  Must resolve within the allowed root.
        """
        # --- dangerous command guard ---
        for pattern in _patterns:
            if pattern.search(command):
                return (
                    f"Error: command matches blocked pattern "
                    f"({pattern.pattern!r})"
                )

        # --- resolve working directory ---
        cwd = str(root_path)
        if workdir is not None:
            try:
                cwd = _resolve_safe_strict(workdir, root_path)
            except ValueError as exc:
                return f"Error: {exc}"

        # --- clamp timeout ---
        _timeout = min(int(timeout or default_timeout), max_timeout)

        # --- resolve bash executable ---
        executable = _resolve_bash()
        if executable is None:
            return _bash_not_found_hint()

        # --- spawn subprocess ---
        try:
            proc = await asyncio.create_subprocess_exec(
                executable,
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                preexec_fn=_preexec_setpgid if sys.platform != "win32" else None,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=_timeout
                )
            except asyncio.TimeoutError:
                _kill_process_group(proc)
                await proc.communicate()
                return (
                    f"Error: command timed out after {_timeout}s. "
                    "Try increasing timeout or simplifying the command."
                )
        except FileNotFoundError:
            return "Error: bash not found on PATH"
        except OSError as exc:
            return f"Error: {exc}"

        # --- decode with raw byte cap ---
        raw = stdout[:max_raw_bytes]
        truncated_wire = len(stdout) > max_raw_bytes
        output = raw.decode("utf-8", errors="replace")

        # --- middle truncation ---
        if truncated_wire or len(output) > max_chars:
            total = len(stdout)
            head_len = max_chars * 4 // 10
            tail_len = max_chars * 4 // 10
            head = output[:head_len]
            tail = output[-tail_len:] if tail_len > 0 else ""
            output = (
                f"{head}\n"
                f"[Truncated: first {head_len} and last {tail_len} "
                f"chars of {total} total chars shown. "
                f"Simplify the command or write output to a file.]\n"
                f"{tail}"
            )

        # --- prepend exit code on failure ---
        exit_code = proc.returncode
        if exit_code != 0:
            output = f"Exit code: {exit_code}\n{output}"

        return output.rstrip("\n")

    return _make_tool_info(bash)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _preexec_setpgid() -> None:
    """Create a new process group so we can kill the whole tree on timeout.

    Used as ``preexec_fn`` argument to ``create_subprocess_exec`` on Unix.
    No-op on Windows (``None`` is passed instead).
    """
    setpgrp = getattr(os, "setpgrp", None)
    if setpgrp is not None:
        setpgrp()


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate the process group, falling back to SIGKILL after 2 s.

    On Windows falls back to ``proc.kill()`` immediately.
    """
    if sys.platform == "win32":
        proc.kill()
        return

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
        return

    # --- async-safe SIGKILL timer ---
    def _force_kill() -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    import threading

    threading.Timer(2.0, _force_kill).start()


def _resolve_bash() -> str | None:
    """Return the path to a usable ``bash`` executable, or ``None``."""
    # 1. Try PATH first
    import shutil

    exe = shutil.which("bash")
    if exe:
        return exe

    # 2. On Windows, probe common Git for Windows install paths
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            str(
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Git"
                / "bin"
                / "bash.exe"
            ),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

    return None


def _bash_not_found_hint() -> str:
    """Return a user-friendly error message when bash cannot be found."""
    if sys.platform == "win32":
        return (
            "Error: bash not found. "
            "Install Git for Windows (winget install Git.Git) or "
            "ensure bash.exe is on PATH."
        )
    return "Error: bash not found on PATH. Install bash (apt install bash, brew install bash, etc.)."
