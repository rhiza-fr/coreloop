"""Tests for make_edit_tool constructor parameters not covered in test_edit_tool.py."""

import tempfile
from pathlib import Path

import pytest

from minimal_agent.tools.edit import make_edit_tool


@pytest.fixture
def sandbox():
    """Create a temp directory as an edit tool root."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


# -- max_bytes rejects oversized files -----------------------------------------


@pytest.mark.asyncio
async def test_max_bytes_rejects_large_file(sandbox):
    """A file exceeding max_bytes returns an error without modifying it."""
    big = Path(sandbox) / "big.txt"
    big.write_text("x" * 200)
    edit = make_edit_tool(sandbox, max_bytes=100)
    result = await edit.fn(path="big.txt", old_text="x", new_text="y")
    assert result.startswith("Error:")
    assert "too large" in result
    assert big.read_text() == "x" * 200  # file unchanged


@pytest.mark.asyncio
async def test_max_bytes_allows_file_within_limit(sandbox):
    """A file within max_bytes is edited normally."""
    f = Path(sandbox) / "small.txt"
    f.write_text("hello world")
    edit = make_edit_tool(sandbox, max_bytes=1000)
    result = await edit.fn(path="small.txt", old_text="world", new_text="there")
    assert "Replaced 1" in result
    assert f.read_text() == "hello there"


# -- new-file creation (old_text="") ------------------------------------------


@pytest.mark.asyncio
async def test_create_new_file(sandbox):
    """old_text='' creates a new file with new_text as content."""
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="new.txt", old_text="", new_text="hello")
    assert "Created" in result
    assert (Path(sandbox) / "new.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_create_file_in_subdirectory(sandbox):
    """Parent directories are created automatically when creating a new file."""
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="sub/dir/new.txt", old_text="", new_text="hi")
    assert "Created" in result
    assert (Path(sandbox) / "sub" / "dir" / "new.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_create_existing_nonempty_file_fails(sandbox):
    """old_text='' on a file that already has content returns an error."""
    f = Path(sandbox) / "existing.txt"
    f.write_text("content")
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="existing.txt", old_text="", new_text="new")
    assert result.startswith("Error:")
    assert f.read_text() == "content"  # unchanged


# -- path traversal for new-file creation -------------------------------------


@pytest.mark.asyncio
async def test_create_file_path_traversal_rejected(sandbox):
    """old_text='' with a traversal path returns an error."""
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="../../escape.txt", old_text="", new_text="x")
    assert result.startswith("Error:")


# -- non-UTF-8 file ------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_non_utf8_file_returns_error(sandbox):
    """A file with non-UTF-8 bytes cannot be edited and returns an error."""
    bad = Path(sandbox) / "bad.bin"
    bad.write_bytes(b"\xff\xfe invalid utf-8")
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="bad.bin", old_text="x", new_text="y")
    assert "non-UTF-8" in result


# -- empty existing file (old_text="" on zero-byte file) ----------------------


@pytest.mark.asyncio
async def test_edit_empty_existing_file_sets_content(sandbox):
    """old_text='' on a zero-byte existing file replaces content with new_text."""
    f = Path(sandbox) / "empty.txt"
    f.write_bytes(b"")
    edit = make_edit_tool(sandbox)
    result = await edit.fn(path="empty.txt", old_text="", new_text="hello")
    assert "Replaced 1" in result
    assert f.read_text() == "hello"
