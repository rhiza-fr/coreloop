"""Tests for make_web_tools: ImportError fallback, web_search, and web_fetch."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreloop.web_tools import make_web_tools


def _mock_client(*, json=None, text=None, headers=None, raise_for_status=None):
    """Return an AsyncClient context-manager mock."""
    response = AsyncMock()
    response.headers = headers or {}
    if json is not None:
        response.json = MagicMock(return_value=json)
    if text is not None:
        response.text = text
    response.raise_for_status = MagicMock(side_effect=raise_for_status)
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=response)
    return client


# -- ImportError when html_to_markdown is absent --------------------------------


def test_make_web_tools_raises_import_error_without_html_to_markdown():
    with patch.dict(sys.modules, {"html_to_markdown": None}):
        with pytest.raises(ImportError, match="html-to-markdown"):
            make_web_tools()


# -- make_web_tools returns two ToolInfo objects --------------------------------


def test_make_web_tools_returns_two_tools():
    tools = make_web_tools()
    assert len(tools) == 2
    assert {t.name for t in tools} == {"web_search", "web_fetch"}


# -- web_search -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_returns_no_results():
    client = _mock_client(json={"results": []})
    with patch("coreloop.web_tools.httpx.AsyncClient", return_value=client):
        tool = next(t for t in make_web_tools(searxng_url="http://localhost:8080") if t.name == "web_search")
        result = await tool.fn(query="test")
    assert result == "No results found."


@pytest.mark.asyncio
async def test_web_search_formats_results():
    client = _mock_client(json={"results": [{
        "title": "Example", "url": "https://example.com",
        "content": "A snippet.", "publishedDate": "2024-01-01",
    }]})
    with patch("coreloop.web_tools.httpx.AsyncClient", return_value=client):
        tool = next(t for t in make_web_tools(searxng_url="http://localhost:8080") if t.name == "web_search")
        result = await tool.fn(query="test")
    assert "Example" in result
    assert "https://example.com" in result
    assert "A snippet." in result
    assert "2024-01-01" in result


@pytest.mark.asyncio
async def test_web_search_returns_error_on_exception():
    with patch("coreloop.web_tools.httpx.AsyncClient", side_effect=RuntimeError("network down")):
        tool = next(t for t in make_web_tools(searxng_url="http://localhost:8080") if t.name == "web_search")
        result = await tool.fn(query="test")
    assert result.startswith("Error:")
    assert "network down" in result


@pytest.mark.asyncio
async def test_web_search_appends_config_hint_for_searxng_error():
    tool = next(t for t in make_web_tools(searxng_url=None) if t.name == "web_search")
    with patch.dict("os.environ", {"SEARXNG_URL": ""}):
        result = await tool.fn(query="test")
    assert "coreloop.toml" in result


# -- web_fetch ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_fetch_returns_content():
    client = _mock_client(text="<h1>Hello world</h1>")
    with patch("coreloop.web_tools.httpx.AsyncClient", return_value=client):
        tool = next(t for t in make_web_tools() if t.name == "web_fetch")
        result = await tool.fn(url="https://example.com")
    assert "Hello world" in result


@pytest.mark.asyncio
async def test_web_fetch_returns_error_on_exception():
    with patch("coreloop.web_tools.httpx.AsyncClient", side_effect=RuntimeError("timeout")):
        tool = next(t for t in make_web_tools() if t.name == "web_fetch")
        result = await tool.fn(url="https://example.com")
    assert result.startswith("Error:")
    assert "timeout" in result
