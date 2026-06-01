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

from minimal_agent.tools.bash import make_bash_tool

_HAS_BASH = shutil.which("bash") is not None
requires_bash = pytest.mark.skipif(not _HAS_BASH, reason="bash not on PATH")


@pytest.fixture
def sandbox():
    with tempfile.TemporaryDirectory() as tmp:
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
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo hello")
    assert result == "hello"


@requires_bash
@pytest.mark.asyncio
async def test_nonzero_exit_is_prefixed(sandbox):
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo oops && exit 3")
    assert result == "Exit code: 3\noops"


@requires_bash
@pytest.mark.asyncio
async def test_stderr_is_captured(sandbox):
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo to-stderr 1>&2")
    assert "to-stderr" in result


@requires_bash
@pytest.mark.asyncio
async def test_timeout_returns_error_not_hang(sandbox):
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="sleep 5", timeout=1)
    assert result.startswith("Error: command timed out after 1s")


# -- workdir scoping ------------------------------------------------------------


@pytest.mark.asyncio
async def test_workdir_outside_root_is_rejected(sandbox):
    bash = make_bash_tool(sandbox)
    result = await bash.fn(command="echo hi", workdir="..")
    assert result.startswith("Error:")
    assert "traversal" in result or "does not exist" in result


@requires_bash
@pytest.mark.asyncio
async def test_workdir_inside_root_is_honoured(sandbox):
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
    bash = make_bash_tool(sandbox, max_chars=200)
    # Emit far more than max_chars of output.
    result = await bash.fn(command="for i in $(seq 1 5000); do echo line$i; done")
    assert "[Truncated:" in result
    assert len(result) < 5000
