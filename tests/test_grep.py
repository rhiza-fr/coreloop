"""Tests for make_grep_tool: path scoping, pattern matching, and output truncation."""

import shutil
import tempfile
from pathlib import Path

import pytest

from minimal_agent.tools.grep import make_grep_tool

_HAS_RG = shutil.which("rg") is not None
requires_rg = pytest.mark.skipif(not _HAS_RG, reason="rg (ripgrep) not on PATH")


@pytest.fixture
def sandbox():
    """Create a temp directory with Python and text files for grep testing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        (tmp_p / "alpha.py").write_text("def hello():\n    return 42\n")
        (tmp_p / "beta.txt").write_text("hello world\ngoodbye world\n")
        sub = tmp_p / "sub"
        sub.mkdir()
        (sub / "gamma.py").write_text("def world():\n    pass\n")
        yield tmp


# -- path-traversal guard -------------------------------------------------------


@pytest.mark.asyncio
async def test_path_traversal_is_rejected(sandbox):
    """A path that escapes the root returns an error."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello", path="..")
    assert result.startswith("Error:")


# -- no-rg fallback -------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_error_when_rg_not_found(sandbox, monkeypatch):
    """A missing rg binary returns a user-friendly error message."""
    monkeypatch.setenv("PATH", "")
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello")
    assert "rg" in result.lower() or result.startswith("Error:")


# -- basic matching -------------------------------------------------------------


@requires_rg
@pytest.mark.asyncio
async def test_finds_pattern_in_files(sandbox):
    """A pattern that exists in the tree is found and returned."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello")
    assert "hello" in result


@requires_rg
@pytest.mark.asyncio
async def test_no_matches_returns_sentinel(sandbox):
    """A pattern with no matches returns the '(no matches)' sentinel."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="zzz_no_such_pattern_xyz")
    assert result == "(no matches)"


@requires_rg
@pytest.mark.asyncio
async def test_file_type_filter(sandbox):
    """file_type restricts matches to files of that type only."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello", file_type="py")
    assert "alpha.py" in result
    assert "beta.txt" not in result


@requires_rg
@pytest.mark.asyncio
async def test_files_with_matches_flag(sandbox):
    """files_with_matches=True returns only file paths, not match lines."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello", files_with_matches=True)
    assert "alpha.py" in result or "beta.txt" in result
    assert "def hello" not in result


@requires_rg
@pytest.mark.asyncio
async def test_negative_after_context_is_rejected(sandbox):
    """after_context < 0 returns an error without running rg."""
    grep = make_grep_tool(sandbox)
    result = await grep.fn(pattern="hello", after_context=-1)
    assert result.startswith("Error:")


# -- truncation -----------------------------------------------------------------


@requires_rg
@pytest.mark.asyncio
async def test_large_output_is_truncated(sandbox):
    """Output exceeding max_chars is truncated with a '...truncated' suffix."""
    tmp_p = Path(sandbox)
    (tmp_p / "big.txt").write_text(("match\n" * 5000))
    grep = make_grep_tool(sandbox, max_chars=200)
    result = await grep.fn(pattern="match")
    assert "truncated" in result
    assert len(result) < 5000
