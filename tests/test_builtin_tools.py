"""Tests for the built-in file tools (read, ls, edit) and path traversal."""

import os
import tempfile
from pathlib import Path

import pytest

from minimal_agent.tools import make_tools
from minimal_agent.tools._shared import _resolve_safe


# -- Fixtures --------------------------------------------------


@pytest.fixture
def sandbox():
    """Create a temp directory with known files and return its path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        (tmp_p / "hello.txt").write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        sub = tmp_p / "sub"
        sub.mkdir(exist_ok=True)
        (sub / "nested.txt").write_text("nested content\n", encoding="utf-8")
        (sub / "other.py").write_text("x = 1\n", encoding="utf-8")
        yield tmp


@pytest.mark.asyncio
async def test_read_full_file(sandbox):
    """Reading a file with no offset/limit returns its entire content."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    result = await read.fn("hello.txt")
    assert result == "line1\nline2\nline3\nline4\nline5\n"


@pytest.mark.asyncio
async def test_read_with_offset(sandbox):
    """An offset skips the first N-1 lines and returns the remainder."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    result = await read.fn("hello.txt", offset=3)
    assert result == "line3\nline4\nline5\n"


@pytest.mark.asyncio
async def test_read_with_offset_and_limit(sandbox):
    """offset + limit returns exactly N lines and appends a truncation hint."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    result = await read.fn("hello.txt", offset=2, limit=2)
    assert (
        result
        == "line2\nline3\n\n[Truncated: showing lines 2-3. Call again with offset=4 to continue.]"
    )


@pytest.mark.asyncio
async def test_read_beyond_end(sandbox):
    """An offset past the last line returns an empty string."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    result = await read.fn("hello.txt", offset=100)
    assert result == ""


@pytest.mark.asyncio
async def test_read_nonexistent_file(sandbox):
    """Reading a missing file returns an Error message."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    result = await read.fn("does_not_exist.txt")
    assert "Error" in result
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_ls_root(sandbox):
    """ls on '.' lists both files and subdirectory entries."""
    tools = {t.name: t for t in make_tools(sandbox)}
    ls = tools["ls"]
    result = await ls.fn(".")
    assert "hello.txt" in result
    assert "sub/" in result


@pytest.mark.asyncio
async def test_ls_subdir(sandbox):
    """ls on a subdirectory lists its contents."""
    tools = {t.name: t for t in make_tools(sandbox)}
    ls = tools["ls"]
    result = await ls.fn("sub")
    assert "nested.txt" in result
    assert "other.py" in result


@pytest.mark.asyncio
async def test_ls_nonexistent(sandbox):
    """ls on a missing path returns an Error message."""
    tools = {t.name: t for t in make_tools(sandbox)}
    ls = tools["ls"]
    result = await ls.fn("does_not_exist")
    assert "Error" in result


@pytest.mark.asyncio
async def test_edit_replace(sandbox):
    """edit replaces a unique string and the change is visible on subsequent read."""
    tools = {t.name: t for t in make_tools(sandbox)}
    edit = tools["edit"]
    result = await edit.fn("hello.txt", old_text="line3", new_text="CHANGED")
    assert "Replaced 1" in result

    # Verify
    tools2 = {t.name: t for t in make_tools(sandbox)}
    read = tools2["read"]
    content = await read.fn("hello.txt")
    assert "CHANGED" in content
    assert "line3" not in content


@pytest.mark.asyncio
async def test_edit_multiple_with_line_hint(sandbox):
    """A duplicate old_text requires line_hint; with it, only that occurrence changes."""
    (Path(sandbox) / "multi.txt").write_text("foo\nfoo\nfoo\n", encoding="utf-8")

    tools = {t.name: t for t in make_tools(sandbox)}
    edit = tools["edit"]
    # Without line_hint -> error
    result = await edit.fn("multi.txt", old_text="foo", new_text="bar")
    assert "appears 3 times" in result
    assert "line_hint" in result

    # With line_hint -> single replace on that line
    result2 = await edit.fn("multi.txt", old_text="foo", new_text="bar", line_hint=2)
    assert "Replaced 1" in result2

    read = tools["read"]
    content = await read.fn("multi.txt")
    assert content == "foo\nbar\nfoo\n"


@pytest.mark.asyncio
async def test_edit_not_found(sandbox):
    """edit with a missing old_text returns a 'not found' error."""
    tools = {t.name: t for t in make_tools(sandbox)}
    edit = tools["edit"]
    result = await edit.fn("hello.txt", old_text="NO_SUCH_TEXT", new_text="x")
    assert "not found" in result


# -- Path traversal protection ---------------------------------


@pytest.mark.asyncio
async def test_path_traversal_rejected(sandbox):
    """Paths that escape the root are denied with a 'path traversal denied' error."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]

    for bad in ("../../../etc/passwd", "/etc/passwd", "..\\..\\..\\windows\\win.ini"):
        result = await read.fn(bad)
        assert "path traversal denied" in result, f"Expected denial for {bad!r}"


@pytest.mark.asyncio
async def test_absolute_path_within_root(sandbox):
    """An absolute path that resolves inside root is accepted."""
    tools = {t.name: t for t in make_tools(sandbox)}
    read = tools["read"]
    abs_path = os.path.join(sandbox, "hello.txt")
    result = await read.fn(abs_path)
    assert result.startswith("line1")


# -- _resolve_safe unit tests ----------------------------------


def test_resolve_safe_within_root(sandbox):
    """A relative path inside root resolves to an absolute path under root."""
    result = _resolve_safe("hello.txt", sandbox)
    assert os.path.isabs(result)
    assert result.startswith(sandbox)


def test_resolve_safe_traversal(sandbox):
    """A traversal path raises ValueError."""
    with pytest.raises(ValueError, match="path traversal denied"):
        _resolve_safe("../../../etc/passwd", sandbox)


def test_resolve_safe_absolute_outside(sandbox):
    """An absolute path outside root raises ValueError."""
    with pytest.raises(ValueError, match="path traversal denied"):
        _resolve_safe("/etc/passwd", sandbox)
