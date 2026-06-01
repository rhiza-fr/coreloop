"""ma: minimal LLM agent CLI."""

import asyncio
import json
import os
from pathlib import Path

import httpx
import typer

from .. import __version__
from .._logging import setup_logging
from ..agent import Agent
from ..hooks import MaxTurnsHook
from ..profiles import _load_merged_profile, config_path, get_config, resolve_profile
from ._run import once, print_http_error, repl
from ._tools import _ALL_TOOL_NAMES, build_tools

app = typer.Typer(
    name="ma",
    invoke_without_command=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)

_P = "Provider"
_T = "Tools"
_A = "Advanced"
_O = "Output"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"coreloop {__version__}")
        raise typer.Exit()


def _ensure_home_config() -> None:
    dst = Path.home() / "coreloop.toml"
    if dst.exists():
        return
    src = config_path()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Originally installed by coreloop from {src}\n"
        "# Edit this file to configure profiles and default tools.\n\n"
    )
    dst.write_text(header + src.read_text())


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # -- run mode ------------------------------------------------------------------
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Run once and print result (non-interactive)"
    ),
    max_turns: int | None = typer.Option(
        None, "--max-turns", "-n", help="Maximum agent loop iterations"
    ),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
    think: bool | None = typer.Option(
        None, "--think/--no-think", help="Set reasoning_effort to medium / none"
    ),
    llm_extra_body: str | None = typer.Option(
        None, "--extra", "-e", help="Extra JSON merged into the API request body"
    ),
    # -- provider ------------------------------------------------------------------
    profile: str = typer.Option(
        "default", "--profile", help="Named profile from ~/coreloop.toml", rich_help_panel=_P
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (overrides profile)",
        envvar="CORELOOP_MODEL",
        rich_help_panel=_P,
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="API base URL (overrides profile)",
        envvar="CORELOOP_BASE_URL",
        rich_help_panel=_P,
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (overrides profile)",
        envvar="CORELOOP_API_KEY",
        rich_help_panel=_P,
    ),
    # -- tools ---------------------------------------------------------------------
    tools: str | None = typer.Option(
        None,
        "--tools",
        metavar="TOOLS",
        help=f"Comma-separated: {', '.join(sorted(_ALL_TOOL_NAMES))}",
        rich_help_panel=_T,
    ),
    root: str | None = typer.Option(
        None, "--root", "-r", help="Allowed root for file tools (default: cwd)", rich_help_panel=_T
    ),
    searxng_url: str | None = typer.Option(
        None,
        "--searxng-url",
        help="SearXNG base URL for web_search/web_fetch",
        envvar="SEARXNG_URL",
        rich_help_panel=_T,
    ),
    # -- advanced ------------------------------------------------------------------
    llm_timeout: float | None = typer.Option(
        None,
        "--llm-timeout",
        "-t",
        help="Asyncio wall-clock timeout per LLM turn (seconds)",
        rich_help_panel=_A,
    ),
    tool_timeout: float | None = typer.Option(
        None, "--tool-timeout", help="Hard timeout per tool call (seconds)", rich_help_panel=_A
    ),
    http_request_timeout: float | None = typer.Option(
        None,
        "--http-request-timeout",
        help="httpx per-chunk read timeout (seconds)",
        rich_help_panel=_A,
    ),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="LLM response cache directory", rich_help_panel=_A
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable LLM response caching", rich_help_panel=_A
    ),
    # -- output --------------------------------------------------------------------
    json_out: bool = typer.Option(
        False, "--json", help="Output all non-partial messages as JSONL", rich_help_panel=_O
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
        metavar="LEVEL",
        rich_help_panel=_O,
    ),
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
        rich_help_panel=_O,
    ),
) -> None:
    """A minimal LLM agent -- interactive REPL or one-shot with [bold]-p PROMPT[/bold].

    On first run, [cyan]~/coreloop.toml[/cyan] is created from the bundled default.
    Edit it to set your provider credentials and default tools.

    [bold]Examples[/bold]
      ma                                              REPL with default profile (Ollama)
      ma --profile openai -p "Hello"                  one-shot with a named profile
      ma -m gpt-4o --tools read,ls,grep --root .      override model and add file tools
      ma --base-url http://localhost:11434/v1 -m qwen  direct endpoint, no config needed
    """
    if ctx.invoked_subcommand is not None:
        return

    if log_level is not None:
        _valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level_upper = log_level.upper()
        if level_upper not in _valid_levels:
            typer.echo(
                f"Error: invalid log level {log_level!r}. "
                f"Valid levels: {', '.join(sorted(_valid_levels))}",
                err=True,
            )
            raise typer.Exit(1)
        setup_logging(level_upper)

    _ensure_home_config()

    try:
        agent_cfg = resolve_profile(profile)
        raw = _load_merged_profile(profile)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    # CLI flags override profile values
    if model is not None:
        agent_cfg.model = model
    if base_url is not None:
        agent_cfg.base_url = base_url
    if api_key is not None:
        agent_cfg.api_key = api_key
    if system is not None:
        agent_cfg.system = system
    if root is not None:
        agent_cfg.root = root
    if llm_timeout is not None:
        agent_cfg.llm_timeout = llm_timeout
    if tool_timeout is not None:
        agent_cfg.tool_timeout = tool_timeout
    if http_request_timeout is not None:
        agent_cfg.http_request_timeout = http_request_timeout
    if no_cache:
        agent_cfg.cache_dir = None
    elif cache_dir is not None:
        agent_cfg.cache_dir = cache_dir

    extra_body: dict | None = None
    if llm_extra_body:
        try:
            extra_body = json.loads(llm_extra_body)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: --extra must be valid JSON: {exc}", err=True)
            raise typer.Exit(1)
    if think is not None:
        extra_body = (extra_body or {}) | {"reasoning_effort": "medium" if think else "none"}
    if extra_body is not None:
        agent_cfg.llm_extra_body = extra_body

    # Resolve tools: --tools flag overrides profile tools list
    tools_opt = tools
    if tools_opt is None and agent_cfg.tools:
        tools_opt = ",".join(agent_cfg.tools)
    if searxng_url is None:
        searxng_url = get_config("tool.web_search.url", raw) or os.environ.get("SEARXNG_URL")
    built_tools = build_tools(tools_opt, agent_cfg.root, searxng_url, raw)

    if max_turns is None:
        max_turns = int(get_config("ui.ma.max_turns", raw, 20))

    try:
        agent = Agent.from_config(agent_cfg, hooks=MaxTurnsHook(max_turns), tools=built_tools)
    except (KeyError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if prompt is not None:
        try:
            asyncio.run(once(agent, prompt, json_out=json_out))
        except httpx.HTTPStatusError as exc:
            print_http_error(exc)
            raise typer.Exit(1)
        except httpx.TimeoutException:
            typer.echo(
                f"Error: {agent_cfg.model} did not respond within {agent_cfg.llm_timeout}s"
                " (increase with --llm-timeout)",
                err=True,
            )
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed: {exc}", err=True)
            raise typer.Exit(1)
    else:
        try:
            asyncio.run(repl(agent, agent_cfg, profile, tools_opt, searxng_url, max_turns, raw))
        except httpx.HTTPStatusError as exc:
            print_http_error(exc)
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed: {exc}", err=True)
            raise typer.Exit(1)
