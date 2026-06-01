"""Tool construction from CLI tool names and profile config."""

from pathlib import Path
from typing import Any

import typer

from ..profiles import get_config
from ..tool_registry import ToolInfo
from ..tools import make_tools

_FILE_TOOL_NAMES = frozenset({"read", "ls", "edit", "grep"})
_WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
_BUILTIN_TOOL_NAMES = _FILE_TOOL_NAMES | {"bash"}
_ALL_TOOL_NAMES = _BUILTIN_TOOL_NAMES | _WEB_TOOL_NAMES


def build_tools(
    tools_opt: str | None,
    root: str | None,
    searxng_url: str | None = None,
    profile: dict[str, Any] | None = None,
) -> list[ToolInfo] | None:
    """Build pre-scoped ToolInfo objects for the given comma-separated tool names.

    Returns None if *tools_opt* is empty (lets the Agent fall back to its own defaults).
    Exits with an error message on unknown tool names.
    """
    if not tools_opt:
        return None
    names = {n.strip().lower() for n in tools_opt.split(",") if n.strip()}
    unknown = names - _ALL_TOOL_NAMES
    if unknown:
        typer.echo(
            f"Unknown tools: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(_ALL_TOOL_NAMES))}",
            err=True,
        )
        raise typer.Exit(1)

    result: list[ToolInfo] = []

    if names & _FILE_TOOL_NAMES:
        fs_tools = make_tools(
            allowed_root=root,
            read_max_lines=get_config("tool.read.max_lines", profile, 100),
            read_max_bytes=get_config("tool.read.max_bytes", profile, 10 * 1024 * 1024),
            ls_max_entries=get_config("tool.ls.max_entries", profile, 500),
            edit_max_bytes=get_config("tool.edit.max_bytes", profile, 10 * 1024 * 1024),
            search_max_chars=get_config("tool.grep.max_chars", profile, 20_000),
            search_timeout=get_config("tool.grep.timeout", profile, 30.0),
        )
        result.extend(t for t in fs_tools if t.name in names)

    if "bash" in names:
        from ..tools.bash import make_bash_tool

        result.append(
            make_bash_tool(
                str(Path(root).resolve()) if root else ".",
                max_chars=get_config("tool.bash.max_chars", profile, 10_000),
                max_raw_bytes=get_config("tool.bash.max_raw_bytes", profile, 100 * 1024),
                default_timeout=get_config("tool.bash.default_timeout", profile, 180),
                max_timeout=get_config("tool.bash.max_timeout", profile, 300),
            )
        )

    if names & _WEB_TOOL_NAMES:
        try:
            from ..web_tools import make_web_tools

            web_tools = make_web_tools(searxng_url=searxng_url)
        except ImportError:
            typer.echo(
                "To use web_search or web_fetch, install web extras: "
                "pip install minimal-agent[web]",
                err=True,
            )
            raise typer.Exit(1)
        result.extend(t for t in web_tools if t.name in names)

    return result
