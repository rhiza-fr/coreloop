"""Thorough unit tests for the built-in ``edit`` tool.

Tests single-replacement-only semantics, empty/error cases, special characters,
concurrency, and large files.
"""

import os
import tempfile
from pathlib import Path

import pytest

from coreloop.tools import make_tools


# -- Fixture ---------------------------------------------------


@pytest.fixture
def sandbox():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hello.txt").write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "nested.txt").write_text("nested content\n", encoding="utf-8")
        (root / "special.txt").write_text(
            "tab\tseparated\nnewline\nhere\nunicode: nout Lhess\n", encoding="utf-8"
        )
        (root / "repeated.txt").write_text("aaa\n", encoding="utf-8")
        (root / "multi.txt").write_text(
            "hello world  \nhello world  \nhello world  \n", encoding="utf-8"
        )
        yield tmp


def _edit(sandbox):
    return {t.name: t for t in make_tools(sandbox)}["edit"]


def _read(sandbox):
    return {t.name: t for t in make_tools(sandbox)}["read"]


# -- Single-replacement (unique match) -------------------------


@pytest.mark.asyncio
async def test_edit_unique(sandbox):
    """Unique old_text -> single replace works."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="line3", new_text="CHANGED")
    assert "Replaced 1" in result

    read = _read(sandbox)
    assert "CHANGED" in await read.fn("hello.txt")
    assert "line3" not in await read.fn("hello.txt")


@pytest.mark.asyncio
async def test_edit_empty_new_text_unique(sandbox):
    """Unique old_text with empty replacement = deletion (leaves a blank line)."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="line3", new_text="")
    assert "Replaced 1" in result

    read = _read(sandbox)
    content = await read.fn("hello.txt")
    # "line3" removed -> line3 becomes empty (blank line between line2 and line4)
    assert content == "line1\nline2\n\nline4\nline5\n"


@pytest.mark.asyncio
async def test_edit_identity_unique(sandbox):
    """Replacing unique text with itself is a no-op."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="line3", new_text="line3")
    assert "Replaced 1" in result

    read = _read(sandbox)
    assert "line3" in await read.fn("hello.txt")


# -- Ambiguous (multiple occurrences) -------------------------


@pytest.mark.asyncio
async def test_edit_ambiguous_errors(sandbox):
    """old_text that appears multiple times -> error."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="line", new_text="ROW")
    assert "appears 5 times" in result


@pytest.mark.asyncio
async def test_edit_ambiguous_trailing_whitespace(sandbox):
    """Multiple matches -> error."""
    edit = _edit(sandbox)
    result = await edit.fn("multi.txt", old_text="hello world", new_text="matched")
    assert "appears 3 times" in result


# -- Empty / not-found cases -----------------------------------


@pytest.mark.asyncio
async def test_edit_not_found(sandbox):
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="NO_SUCH_TEXT", new_text="x")
    assert "not found" in result


@pytest.mark.asyncio
async def test_edit_empty_old_text(sandbox):
    """Empty old_text is rejected (would insert everywhere)."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="", new_text="X")
    assert "Error" in result
    assert "already has content" in result


# -- Special characters ----------------------------------------


@pytest.mark.asyncio
async def test_edit_with_newlines(sandbox):
    """old_text spanning multiple lines (unique) -> works."""
    edit = _edit(sandbox)
    result = await edit.fn("hello.txt", old_text="line3\nline4", new_text="joined")
    assert "Replaced 1" in result

    read = _read(sandbox)
    content = await read.fn("hello.txt")
    assert "joined\nline5" in content


@pytest.mark.asyncio
async def test_edit_with_tabs(sandbox):
    """Tab character in old_text (unique) -> works."""
    edit = _edit(sandbox)
    result = await edit.fn("special.txt", old_text="\t", new_text="---TAB---")
    assert "Replaced 1" in result

    read = _read(sandbox)
    content = await read.fn("special.txt")
    assert "tab---TAB---separated" in content


@pytest.mark.asyncio
async def test_edit_unicode(sandbox):
    """Unicode old_text (unique) -> works."""
    edit = _edit(sandbox)
    result = await edit.fn("special.txt", old_text="nout Lhess", new_text="replaced")
    assert "Replaced 1" in result

    read = _read(sandbox)
    content = await read.fn("special.txt")
    assert "replaced" in content
    assert "nout Lhess" not in content


# -- Edge cases ------------------------------------------------


@pytest.mark.asyncio
async def test_edit_overlapping(sandbox):
    """Overlapping: 'aa' in 'aaa' -> single non-overlapping match."""
    edit = _edit(sandbox)
    result = await edit.fn("repeated.txt", old_text="aa", new_text="a")
    assert "Replaced 1" in result

    read = _read(sandbox)
    assert await read.fn("repeated.txt") == "aa\n"


@pytest.mark.asyncio
async def test_edit_in_subdirectory(sandbox):
    edit = _edit(sandbox)
    result = await edit.fn("sub/nested.txt", old_text="nested", new_text="CHANGED")
    assert "Replaced 1" in result

    read = _read(sandbox)
    assert "CHANGED content" in await read.fn("sub/nested.txt")


# -- Error / edge conditions -----------------------------------


@pytest.mark.asyncio
async def test_edit_nonexistent_file(sandbox):
    edit = _edit(sandbox)
    result = await edit.fn("no_such_file.txt", old_text="x", new_text="y")
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_edit_path_traversal(sandbox):
    edit = _edit(sandbox)
    for bad in ("../../../outside/secret.txt", "/outside/secret.txt", "..\\..\\..\\outside\\secret.txt"):
        result = await edit.fn(bad, old_text="x", new_text="y")
        assert "path traversal denied" in result, f"Expected denial for {bad!r}"


@pytest.mark.asyncio
async def test_edit_absolute_within_root(sandbox):
    edit = _edit(sandbox)
    abs_path = os.path.join(sandbox, "hello.txt")
    result = await edit.fn(abs_path, old_text="line3", new_text="absolute")
    assert "Replaced" in result


# -- Round-trip: edit then read back ---------------------------


@pytest.mark.asyncio
async def test_edit_round_trip_first_and_last(sandbox):
    """Edit first and last line (both unique after prior edit)."""
    edit = _edit(sandbox)
    await edit.fn("hello.txt", old_text="line1", new_text="FIRST")
    await edit.fn("hello.txt", old_text="line5", new_text="LAST")

    read = _read(sandbox)
    content = await read.fn("hello.txt")
    assert content == "FIRST\nline2\nline3\nline4\nLAST\n"


# -- Concurrency / isolation -----------------------------------


@pytest.mark.asyncio
async def test_edit_concurrent_different_files(sandbox):
    """Edits to different files (unique texts) run safely."""
    edit = _edit(sandbox)
    (Path(sandbox) / "a.txt").write_text("hello\n", encoding="utf-8")
    (Path(sandbox) / "b.txt").write_text("world\n", encoding="utf-8")

    import asyncio

    r1, r2 = await asyncio.gather(
        edit.fn("a.txt", old_text="hello", new_text="HELLO"),
        edit.fn("b.txt", old_text="world", new_text="WORLD"),
    )
    assert "Replaced" in r1
    assert "Replaced" in r2

    read = _read(sandbox)
    assert await read.fn("a.txt") == "HELLO\n"
    assert await read.fn("b.txt") == "WORLD\n"


@pytest.mark.asyncio
async def test_edit_sequential_same_file(sandbox):
    """Sequential edits to unique texts in the same file."""
    edit = _edit(sandbox)
    assert "Replaced" in await edit.fn("hello.txt", old_text="line1", new_text="ONE")
    assert "Replaced" in await edit.fn("hello.txt", old_text="line2", new_text="TWO")

    read = _read(sandbox)
    content = await read.fn("hello.txt")
    assert "ONE" in content and "TWO" in content


# -- Large file ------------------------------------------------


@pytest.mark.asyncio
async def test_edit_large_file_unique(sandbox):
    """Unique edit in a 10 000-line file."""
    large = Path(sandbox) / "large.txt"
    large.write_text(
        "\n".join(f"line_{i}" for i in range(10000)) + "\n",
        encoding="utf-8",
    )

    edit = _edit(sandbox)
    result = await edit.fn("large.txt", old_text="line_5000", new_text="CHANGED")
    assert "Replaced 1" in result

    read = _read(sandbox)
    assert "CHANGED" in await read.fn("large.txt", offset=5001, limit=1)
