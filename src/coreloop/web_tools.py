"""Optional web tools: ``web_search`` and ``web_fetch``.

Requires the ``web`` extra: ``pip install coreloop[web]``
"""

import html as _html
import os
import re
from typing import Literal

import httpx

from .tool_registry import ToolInfo
from .tools._shared import _make_tool_info

_DEFAULT_TIMEOUT = 15.0
_MAX_CONTENT_BYTES = 1_000_000
_MAX_MARKDOWN_CHARS = 100_000
_MAX_RAW_CHARS = 50_000


def _searxng_base(override: str | None) -> str:
    url = override or os.environ.get("SEARXNG_URL", "")
    if not url:
        raise RuntimeError("SearXNG URL not configured (set SEARXNG_URL env var)")
    return url.rstrip("/")


def _metadata(html_content: str) -> str:
    meta: dict[str, str] = {}
    if m := re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL):
        meta["title"] = _html.unescape(m.group(1).strip())
    if m := re.search(
        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
        html_content, re.IGNORECASE,
    ):
        meta["description"] = _html.unescape(m.group(1).strip())
    for prop, value in re.findall(
        r'<meta[^>]*property=["\']og:(\w+)["\'][^>]*content=["\']([^"\']*)["\']',
        html_content, re.IGNORECASE,
    ):
        meta[f"og_{prop}"] = _html.unescape(value.strip())
    return "\n".join(f"{k}: {v}" for k, v in meta.items())


def make_web_tools(searxng_url: str | None = None) -> list[ToolInfo]:
    """Build web search and fetch tools.

    Parameters
    ----------
    searxng_url:
        Base URL of your SearXNG instance. Falls back to ``SEARXNG_URL`` env var.
    """
    try:
        from html_to_markdown import convert as _convert
    except ImportError as exc:
        raise ImportError(
            "Web tools require 'html-to-markdown'. Install with: pip install coreloop[web]"
        ) from exc

    async def web_search(
        query: str,
        max_results: int = 5,
        domain_filter: str | None = None,
        recency: Literal["all_time", "day", "week", "month", "year"] = "all_time",
    ) -> str:
        """Use this to find information on the web. Returns titles, URLs, and snippets for relevant pages.

        Parameters
        ----------
        query:
            Search query string.
        max_results:
            Maximum number of results to return.
        domain_filter:
            Restrict results to a specific domain (e.g. ``"github.com"``).
        recency:
            Time filter: ``all_time``, ``day``, ``week``, ``month``, or ``year``.
        """
        try:
            base = _searxng_base(searxng_url)
            q = f"site:{domain_filter} {query}" if domain_filter else query
            params: dict[str, str | int] = {"q": q, "format": "json", "categories": "general"}
            if recency != "all_time":
                params["time_range"] = recency
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                r = await client.get(f"{base}/search", params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            msg = str(exc)
            if "SearXNG URL not configured" in msg:
                msg += " or set searxng_url in ~/coreloop.toml"
            return f"Error: {msg}"

        results = data.get("results", [])[:max_results]
        if not results:
            return "No results found."

        lines: list[str] = []
        for item in results:
            lines.append(f"**{item.get('title', '')}**")
            lines.append(item.get("url", ""))
            if snippet := item.get("content", ""):
                lines.append(snippet)
            if date := item.get("publishedDate"):
                lines.append(f"Published: {date}")
            lines.append("")
        return "\n".join(lines).rstrip()

    async def web_fetch(
        url: str,
        extract_mode: Literal["markdown", "raw", "metadata"] = "markdown",
    ) -> str:
        """Use this to read a specific web page when you already have its URL.

        Parameters
        ----------
        url:
            HTTP/HTTPS URL to fetch.
        extract_mode:
            How to extract content: ``markdown`` (default), ``raw`` (HTML), or ``metadata``.
        """
        try:
            async with httpx.AsyncClient(
                timeout=_DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "coreloop/web (https://github.com/rhiza-fr/coreloop)"},
            ) as client:
                r = await client.get(url)
                r.raise_for_status()
                if cl := r.headers.get("content-length"):
                    if int(cl) > _MAX_CONTENT_BYTES:
                        return f"Error: Content too large ({cl} bytes)"
                page_html = r.text
        except Exception as exc:
            return f"Error: {exc}"

        if extract_mode == "raw":
            return page_html[:_MAX_RAW_CHARS]
        if extract_mode == "metadata":
            return _metadata(page_html)
        return _convert(page_html).content[:_MAX_MARKDOWN_CHARS]

    return [_make_tool_info(web_search), _make_tool_info(web_fetch)]
