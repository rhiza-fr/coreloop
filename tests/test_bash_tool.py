"""Tests for the bash tool: execution, exit-code reporting, the dangerous-pattern
guard, workdir scoping, and output truncation.

Safety note: the dangerous-pattern tests assert the guard *blocks* the command.
The guard returns before any subprocess is spawned, so no destructive command is
ever executed -- these tests cannot harm the machine.
"""

import os
import shutil
import tempfile

import pytest

from coreloop.tools.bash import make_bash_tool

_HAS_BASH = shutil.which("bash") is not None
requires_bash = pytest.mark.skipif(not _HAS_BASH, reason="bash not on PATH")


@pytest.fixture
def sandbox():
    """Create a temp directory as a bash tool root."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        yield tmp


# -- the guard (no subprocess spawned; safe regardless of bash presence) --------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf / --no-preserve-root",
        "rm -fr /",
        "rm -rf /etc",
        "rm -rf /home",
        "rm -rf /tmp/",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
        "chmod 777 /",
    ],
)
async def test_dangerous_command_is_blocked(sandbox, command):
    """Each pattern in the default blocklist must reject the matching command."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command=command)
    assert result.startswith("Error: command matches blocked pattern")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/build",  # deep path -- ordinary recursive delete
        "rm -rf ./build",
        "echo /",
        "ls -la",
    ],
)
async def test_safe_command_not_blocked(sandbox, command):
    """These must NOT trip the guard. We only check the guard verdict, so for
    commands that would still run we assert merely that no block error fired."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command=command)
    assert not result.startswith("Error: command matches blocked pattern")


# -- execution ------------------------------------------------------------------


@requires_bash
@pytest.mark.asyncio
async def test_executes_and_returns_stdout(sandbox):
    """A simple echo command returns its stdout."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo hello")
    assert result == "hello"


@requires_bash
@pytest.mark.asyncio
async def test_nonzero_exit_is_prefixed(sandbox):
    """Non-zero exit code is prepended as 'Exit code: N'."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo oops && exit 3")
    assert result == "Exit code: 3\noops"


@requires_bash
@pytest.mark.asyncio
async def test_stderr_is_captured(sandbox):
    """stderr is merged into stdout and returned."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo to-stderr 1>&2")
    assert "to-stderr" in result


@requires_bash
@pytest.mark.asyncio
async def test_timeout_returns_error_not_hang(sandbox):
    """A command that exceeds the timeout returns an error, not a hang."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="sleep 5", timeout=1)
    assert result.startswith("Error: command timed out after 1s")


# -- workdir scoping ------------------------------------------------------------


@pytest.mark.asyncio
async def test_workdir_outside_root_is_rejected(sandbox):
    """A workdir that escapes root is rejected with an error."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo hi", workdir="..")
    assert result.startswith("Error:")
    assert "traversal" in result or "does not exist" in result


@requires_bash
@pytest.mark.asyncio
async def test_workdir_inside_root_is_honoured(sandbox):
    """A workdir inside root is used as the command's cwd."""
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="pwd", workdir=".")
    # Git bash on Windows reports MSYS-style paths, so assert on the leaf dir
    # name, which survives the path translation on every platform.
    leaf = os.path.basename(sandbox.rstrip("/\\"))
    assert leaf in result


# -- truncation -----------------------------------------------------------------


@requires_bash
@pytest.mark.asyncio
async def test_large_output_is_truncated(sandbox):
    """Output exceeding max_chars is middle-truncated with a [Truncated:] marker."""
    bash = make_bash_tool(sandbox, max_chars=200)
    # Emit far more than max_chars of output.
    result = await bash.fn(command="for ((i=1; i<=5000; i++)); do echo line; done")
    assert "[Truncated:" in result
    assert len(result) < 5000
