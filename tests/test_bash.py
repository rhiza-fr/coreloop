"""Tests for make_bash_tool constructor parameters not covered in test_bash_tool.py."""

import shutil
import tempfile

import pytest

from minimal_agent.tools.bash import make_bash_tool

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
