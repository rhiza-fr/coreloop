"""Optional web tools: ``web_search`` and ``web_fetch``.

Requires the ``web`` extra: ``pip install minimal-agent[web]``
"""

from .registry import ToolInfo
from .tools._shared import _make_tool_info


def make_web_tools(searxng_url: str | None = None) -> list[ToolInfo]:
    """Build web search and fetch tools.

    Parameters
    ----------
    searxng_url:
        Base URL of your SearXNG instance (e.g. ``"http://localhost:8080"``).
        Falls back to the ``SEARXNG_URL`` environment variable or pvl-webtools'
        own default if ``None``.
    """
    try:
        from pvlwebtools import web_fetch as _web_fetch
        from pvlwebtools import web_search as _web_search
    except ImportError as exc:
        raise ImportError(
            "Web tools require 'pvl-webtools'. "
            "Install with: pip install minimal-agent[web]"
        ) from exc

    async def web_search(
        query: str,
        max_results: int = 5,
        domain_filter: str | None = None,
        recency: str = "all_time",
    ) -> str:
        """Search the web via SearXNG and return titles, URLs, and snippets.

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
            if max_results is not None:
                max_results = int(max_results)
            results = await _web_search(
                query=query,
                max_results=max_results,
                domain_filter=domain_filter or None,
                recency=recency,  # type: ignore[arg-type]
                searxng_url=searxng_url,
            )
        except Exception as exc:
            msg = str(exc)
            if "SearXNG URL not configured" in msg:
                msg += " or set searxng_url in ~/.ma-config.toml"
            return f"Error: {msg}"

        if not results:
            return "No results found."

        lines: list[str] = []
        for r in results:
            lines.append(f"**{r.title}**")
            lines.append(r.url)
            if r.snippet:
                lines.append(r.snippet)
            if r.published_date:
                lines.append(f"Published: {r.published_date}")
            lines.append("")
        return "\n".join(lines).rstrip()

    async def web_fetch(
        url: str,
        extract_mode: str = "markdown",
    ) -> str:
        """Fetch a web page and return its content as markdown (or raw text).

        Parameters
        ----------
        url:
            HTTP/HTTPS URL to fetch.
        extract_mode:
            How to extract content: ``markdown`` (full page as markdown),
            ``article`` (main article text only), ``raw`` (raw HTML),
            or ``metadata`` (title, description, etc.).
        """
        try:
            result = await _web_fetch(url=url, extract_mode=extract_mode)  # type: ignore[arg-type]
        except Exception as exc:
            return f"Error: {exc}"
        return result.content

    return [_make_tool_info(web_search), _make_tool_info(web_fetch)]
