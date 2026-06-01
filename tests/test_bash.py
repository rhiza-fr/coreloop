"""Tests for make_bash_tool constructor parameters not covered in test_bash_tool.py."""

import shutil
import tempfile
from unittest.mock import patch

import pytest

from coreloop.tools.bash import make_bash_tool

_HAS_BASH = shutil.which("bash") is not None
requires_bash = pytest.mark.skipif(not _HAS_BASH, reason="bash not on PATH")


@pytest.fixture
def sandbox():
    """Create a temp directory as a bash tool root."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        yield tmp


# -- custom dangerous_patterns --------------------------------------------------


@pytest.mark.asyncio
async def test_custom_dangerous_patterns_blocks_matching_command(sandbox):
    """A command matching a custom pattern is blocked."""
    bash = make_bash_tool(sandbox, dangerous_patterns=[r"\bdrop table\b"])
    result = await bash.fn(command="drop table users")
    assert result.startswith("Error: command matches blocked pattern")


@pytest.mark.asyncio
async def test_custom_dangerous_patterns_replaces_defaults(sandbox):
    """Passing dangerous_patterns replaces (not extends) the default list."""
    bash = make_bash_tool(sandbox, dangerous_patterns=[r"\bdrop table\b"])
    # rm -rf / is in the default list but not in our custom list -- must not be blocked
    result = await bash.fn(command="rm -rf /etc")
    assert not result.startswith("Error: command matches blocked pattern")


@pytest.mark.asyncio
async def test_empty_dangerous_patterns_blocks_nothing(sandbox):
    """An empty dangerous_patterns list disables all blocking."""
    bash = make_bash_tool(sandbox, dangerous_patterns=[])
    result = await bash.fn(command="rm -rf /")
    assert not result.startswith("Error: command matches blocked pattern")


# -- max_timeout clamps user-supplied timeout -----------------------------------


@requires_bash
@pytest.mark.asyncio
async def test_max_timeout_clamps_requested_timeout(sandbox):
    """A timeout larger than max_timeout is silently clamped to max_timeout."""
    bash = make_bash_tool(sandbox, max_timeout=1)
    result = await bash.fn(command="sleep 5", timeout=9999)
    # Should time out at 1s, not 9999s
    assert result.startswith("Error: command timed out after 1s")


# -- max_raw_bytes truncates raw output before decoding -------------------------


@requires_bash
@pytest.mark.asyncio
async def test_max_raw_bytes_triggers_truncation(sandbox):
    """Output exceeding max_raw_bytes is truncated even if under max_chars."""
    bash = make_bash_tool(sandbox, max_raw_bytes=50, max_chars=10_000)
    result = await bash.fn(command="python3 -c \"print('x' * 5000)\"")
    assert "[Truncated:" in result


# -- default_timeout is used when no timeout arg is passed ----------------------


@requires_bash
@pytest.mark.asyncio
async def test_default_timeout_is_applied(sandbox):
    """default_timeout is used when the caller does not pass a timeout argument."""
    bash = make_bash_tool(sandbox, default_timeout=1, max_timeout=2)
    result = await bash.fn(command="sleep 5")
    assert result.startswith("Error: command timed out after 1s")


# -- max_chars truncates decoded output ----------------------------------------


@requires_bash
@pytest.mark.asyncio
async def test_max_chars_triggers_truncation(sandbox):
    """Output exceeding max_chars is middle-truncated with a [Truncated:] marker."""
    bash = make_bash_tool(sandbox, max_chars=100)
    result = await bash.fn(command="for ((i=1; i<=500; i++)); do echo line; done")
    assert "[Truncated:" in result
    assert len(result) < 5000


# -- bash not found ------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_not_found_returns_error(sandbox):
    """When no bash executable is on PATH, the tool returns a user-friendly error."""
    with patch("coreloop.tools.bash._resolve_bash", return_value=None):
        bash = make_bash_tool(sandbox)
        result = await bash.fn(command="echo hi")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_subprocess_file_not_found_returns_error(sandbox):
    """FileNotFoundError during subprocess spawn is caught and returned as an error."""
    async def _raise(*args, **kwargs):
        raise FileNotFoundError("no such file")

    with patch("asyncio.create_subprocess_exec", side_effect=_raise):
        bash = make_bash_tool(sandbox)
        result = await bash.fn(command="echo hi")
    assert "bash not found" in result


@pytest.mark.asyncio
async def test_subprocess_os_error_returns_error(sandbox):
    """OSError during subprocess spawn is caught and returned as an error."""
    async def _raise(*args, **kwargs):
        raise OSError("permission denied")

    with patch("asyncio.create_subprocess_exec", side_effect=_raise):
        bash = make_bash_tool(sandbox)
        result = await bash.fn(command="echo hi")
    assert result.startswith("Error:")


# -- _preexec_setpgid ----------------------------------------------------------


def test_preexec_setpgid_calls_setpgrp_when_available():
    """_preexec_setpgid calls os.setpgrp() when it exists on the platform."""
    from unittest.mock import MagicMock
    from coreloop.tools.bash import _preexec_setpgid

    mock_setpgrp = MagicMock()
    with patch("os.setpgrp", mock_setpgrp, create=True):
        _preexec_setpgid()
    mock_setpgrp.assert_called_once()


def test_kill_process_group_windows_calls_proc_kill():
    """On Windows the function falls back to proc.kill() directly."""
    from unittest.mock import MagicMock
    from coreloop.tools.bash import _kill_process_group

    proc = MagicMock()
    with patch("sys.platform", "win32"):
        _kill_process_group(proc)
    proc.kill.assert_called_once()


def test_kill_process_group_unix_sends_sigterm():
    """On Unix the function sends SIGTERM to the process group."""
    import signal as _signal
    from unittest.mock import MagicMock
    from coreloop.tools.bash import _kill_process_group

    proc = MagicMock()
    proc.pid = 1234
    with (
        patch("sys.platform", "linux"),
        patch("os.getpgid", return_value=5678, create=True),
        patch("os.killpg", create=True) as mock_killpg,
        patch("threading.Timer") as mock_timer,
    ):
        _kill_process_group(proc)
    mock_killpg.assert_called_once_with(5678, _signal.SIGTERM)
    mock_timer.assert_called_once()


def test_kill_process_group_unix_falls_back_on_os_error():
    """An OSError from os.getpgid causes fallback to proc.kill()."""
    from unittest.mock import MagicMock
    from coreloop.tools.bash import _kill_process_group

    proc = MagicMock()
    proc.pid = 1234
    with (
        patch("sys.platform", "linux"),
        patch("os.getpgid", side_effect=OSError("no such process"), create=True),
    ):
        _kill_process_group(proc)
    proc.kill.assert_called_once()


def test_preexec_setpgid_no_op_when_setpgrp_absent():
    """_preexec_setpgid does nothing when os.setpgrp is absent (Windows)."""
    from coreloop.tools.bash import _preexec_setpgid
    import os as _os

    original = getattr(_os, "setpgrp", None)
    try:
        if hasattr(_os, "setpgrp"):
            delattr(_os, "setpgrp")
        _preexec_setpgid()  # must not raise
    finally:
        if original is not None:
            setattr(_os, "setpgrp", original)
