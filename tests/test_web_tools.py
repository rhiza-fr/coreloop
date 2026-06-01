"""Tests for make_web_tools: ImportError fallback, web_search, and web_fetch."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from coreloop.web_tools import make_web_tools


def _fake_pvlwebtools(search_return=None, fetch_return=None, search_exc=None, fetch_exc=None):
    """Build a SimpleNamespace that mimics the pvlwebtools module for testing."""
    mod = SimpleNamespace()

    async def web_search(**kwargs):
        if search_exc:
            raise search_exc
        return search_return or []

    async def web_fetch(**kwargs):
        if fetch_exc:
            raise fetch_exc
        return fetch_return

    mod.web_search = web_search
    mod.web_fetch = web_fetch
    return mod


@pytest.fixture(autouse=True)
def _patch_pvlwebtools(request):
    """Each test can set request.param to a fake module; default is the real one."""
    fake = getattr(request, "param", None)
    if fake is None:
        yield
        return
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        yield


# -- ImportError when pvlwebtools is absent --------------------------------------


def test_make_web_tools_raises_import_error_without_pvlwebtools():
    """make_web_tools raises ImportError with install hint when pvlwebtools is absent."""
    with patch.dict(sys.modules, {"pvlwebtools": None}):
        with pytest.raises(ImportError, match="pvl-webtools"):
            make_web_tools()


# -- make_web_tools returns two ToolInfo objects ---------------------------------


def test_make_web_tools_returns_two_tools():
    """make_web_tools returns exactly two tools: web_search and web_fetch."""
    tools = make_web_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"web_search", "web_fetch"}


# -- web_search -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_returns_no_results():
    """An empty result list produces the 'No results found.' sentinel."""
    fake = _fake_pvlwebtools(search_return=[])
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_search")
    result = await tool.fn(query="test")
    assert result == "No results found."


@pytest.mark.asyncio
async def test_web_search_formats_results():
    """Search results are formatted with title, URL, snippet, and date."""
    result_obj = MagicMock()
    result_obj.title = "Example"
    result_obj.url = "https://example.com"
    result_obj.snippet = "A snippet."
    result_obj.published_date = "2024-01-01"

    fake = _fake_pvlwebtools(search_return=[result_obj])
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_search")
    result = await tool.fn(query="test")
    assert "Example" in result
    assert "https://example.com" in result
    assert "A snippet." in result
    assert "2024-01-01" in result


@pytest.mark.asyncio
async def test_web_search_returns_error_on_exception():
    """A search exception is caught and returned as an 'Error:' string."""
    fake = _fake_pvlwebtools(search_exc=RuntimeError("network down"))
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_search")
    result = await tool.fn(query="test")
    assert result.startswith("Error:")
    assert "network down" in result


@pytest.mark.asyncio
async def test_web_search_appends_config_hint_for_searxng_error():
    """A SearXNG-not-configured error includes a hint about coreloop.toml."""
    fake = _fake_pvlwebtools(search_exc=RuntimeError("SearXNG URL not configured"))
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_search")
    result = await tool.fn(query="test")
    assert "coreloop.toml" in result


# -- web_fetch ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_fetch_returns_content():
    """web_fetch returns the content string from the FetchResult."""
    fetch_result = MagicMock()
    fetch_result.content = "# Hello world"

    fake = _fake_pvlwebtools(fetch_return=fetch_result)
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_fetch")
    result = await tool.fn(url="https://example.com")
    assert result == "# Hello world"


@pytest.mark.asyncio
async def test_web_fetch_returns_error_on_exception():
    """A fetch exception is caught and returned as an 'Error:' string."""
    fake = _fake_pvlwebtools(fetch_exc=RuntimeError("timeout"))
    with patch.dict(sys.modules, {"pvlwebtools": fake}):
        tools = make_web_tools()
    tool = next(t for t in tools if t.name == "web_fetch")
    result = await tool.fn(url="https://example.com")
    assert result.startswith("Error:")
    assert "timeout" in result
