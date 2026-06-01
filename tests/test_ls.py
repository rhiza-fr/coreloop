"""Tests for make_ls_tool constructor parameters not covered in test_builtin_tools.py."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from coreloop.tools.ls import make_ls_tool


@pytest.fixture
def sandbox():
    """Create a temp directory with files for ls testing."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        for i in range(5):
            (tmp_p / f"file{i}.txt").write_text("x")
        yield tmp


# -- max_entries caps the listing ----------------------------------------------


@pytest.mark.asyncio
async def test_max_entries_caps_listing(sandbox):
    """Entries beyond max_entries are omitted with a '... (N more)' notice."""
    ls = make_ls_tool(sandbox, max_entries=2)
    result = await ls.fn(".")
    lines = result.splitlines()
    assert any("more" in line for line in lines)
    assert sum(1 for line in lines if "file" in line) == 2


@pytest.mark.asyncio
async def test_max_entries_no_truncation_when_under_limit(sandbox):
    """When entries fit within max_entries no truncation notice is added."""
    ls = make_ls_tool(sandbox, max_entries=100)
    result = await ls.fn(".")
    assert "more" not in result
    assert result.count("file") == 5


# -- error paths ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_traversal_rejected(sandbox):
    """A path escaping root returns an error."""
    ls = make_ls_tool(sandbox)
    result = await ls.fn("..")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_not_a_directory_returns_error(sandbox):
    """Listing a file path (not a directory) returns an error."""
    ls = make_ls_tool(sandbox)
    result = await ls.fn("file0.txt")
    assert "not a directory" in result


@pytest.mark.asyncio
async def test_empty_directory_returns_sentinel(sandbox):
    """An empty subdirectory returns the '(empty directory)' sentinel."""
    (Path(sandbox) / "emptydir").mkdir()
    ls = make_ls_tool(sandbox)
    result = await ls.fn("emptydir")
    assert result == "(empty directory)"


@pytest.mark.asyncio
async def test_listdir_os_error_returns_error(sandbox):
    """An OSError from os.listdir is caught and returned as an error."""
    ls = make_ls_tool(sandbox)
    with patch("os.listdir", side_effect=OSError("permission denied")):
        result = await ls.fn(".")
    assert result.startswith("Error:")
    assert "cannot list" in result


@pytest.mark.asyncio
async def test_entry_os_error_shows_question_mark(sandbox):
    """An OSError when stat-ing an entry shows '(?)' instead of crashing."""
    from pathlib import Path as _Path

    call_count = 0

    def _raise_on_second_is_symlink(self):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise OSError("permission denied")
        return False

    with patch.object(_Path, "is_symlink", _raise_on_second_is_symlink):
        ls = make_ls_tool(sandbox)
        result = await ls.fn(".")
    assert "(?)" in result


@pytest.mark.asyncio
async def test_symlink_entry_shown_with_arrow(sandbox):
    """A symlink entry inside root is shown as 'name -> target'."""
    target_path = Path(sandbox) / "file0.txt"

    with (
        patch("os.readlink", return_value=str(target_path), create=True),
        patch("pathlib.Path.is_symlink", lambda self: str(self).endswith("file0.txt")),
    ):
        ls = make_ls_tool(sandbox)
        result = await ls.fn(".")
    assert "->" in result


