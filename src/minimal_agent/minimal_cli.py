"""ma: minimal LLM agent CLI."""

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import httpx
import typer
from rich.console import Console

from . import __version__
from ._logging import setup_logging
from .agent import Agent
from .config import AgentConfig
from .hooks import MaxTurnsHook
from .profiles import _load_merged_profile, config_path, get_config, resolve_profile
from .registry import ToolInfo
from .tools import make_tools
from .types import Message

_console = Console()
_HTTP_ERROR_BODY_PREVIEW = 500
_TOOL_RESULT_PREVIEW = 300

_BUILTIN_TOOL_NAMES = frozenset({"read", "ls", "edit", "grep", "bash"})
_WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
_ALL_TOOL_NAMES = _BUILTIN_TOOL_NAMES | _WEB_TOOL_NAMES

_HTTP_HINTS: dict[int, str] = {
    401: "Authentication failed. Check your API key.",
    403: "Access denied. Your API key may lack permissions for this resource.",
    404: "Endpoint not found. Check the base URL and model name.",
    429: "Rate limited. Wait before retrying, or reduce request frequency.",
}

app = typer.Typer(
    name="ma",
    invoke_without_command=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _print_http_error(exc: httpx.HTTPStatusError) -> None:
    status = exc.response.status_code
    typer.echo(f"Error: HTTP {status} from {exc.request.url}", err=True)
    body = exc.response.text[:_HTTP_ERROR_BODY_PREVIEW].strip()
    if body:
        typer.echo(f"  {body}", err=True)
    hint = _HTTP_HINTS.get(status) or (
        "Server error. The provider may be temporarily unavailable." if status >= 500 else None
    )
    if hint:
        typer.echo(f"  Hint: {hint}", err=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"minimal-agent {__version__}")
        raise typer.Exit()


def _ensure_home_config() -> None:
    dst = Path.home() / "minimal-agent.toml"
    if dst.exists():
        return
    src = config_path()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Originally installed by minimal-agent from {src}\n"
        "# Edit this file to configure profiles and default tools.\n\n"
    )
    dst.write_text(header + src.read_text())


def _build_tools(
    tools_opt: str | None,
    root: str | None,
    searxng_url: str | None = None,
    profile: dict | None = None,
) -> list[ToolInfo] | None:
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
    _FS_TOOL_NAMES = _BUILTIN_TOOL_NAMES - {"bash"}
    if names & _FS_TOOL_NAMES:
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
        from .tools.bash import make_bash_tool

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
        url = searxng_url or os.environ.get("SEARXNG_URL")
        try:
            from .web_tools import make_web_tools

            web_tools = make_web_tools(searxng_url=url)
        except ImportError:
            typer.echo(
                "To use web_search or web_fetch, install web extras: "
                "pip install minimal-agent[web]",
                err=True,
            )
            raise typer.Exit(1)
        result.extend(t for t in web_tools if t.name in names)
    return result


_P = "Provider"
_T = "Tools"
_A = "Advanced"
_O = "Output"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # -- run mode --------------------------------------------------------------
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
    # -- provider --------------------------------------------------------------
    profile: str = typer.Option(
        "default", "--profile", help="Named profile from ~/minimal-agent.toml", rich_help_panel=_P
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (overrides profile)",
        envvar="MINIMAL_AGENT_MODEL",
        rich_help_panel=_P,
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="API base URL (overrides profile)",
        envvar="MINIMAL_AGENT_BASE_URL",
        rich_help_panel=_P,
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (overrides profile)",
        envvar="MINIMAL_AGENT_API_KEY",
        rich_help_panel=_P,
    ),
    # -- tools -----------------------------------------------------------------
    tools: str | None = typer.Option(
        None,
        "--tools",
        metavar="TOOLS",
        help="Comma-separated: read,ls,edit,grep,bash,web_search,web_fetch",
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
    # -- advanced --------------------------------------------------------------
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
    # -- output ----------------------------------------------------------------
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

    On first run, [cyan]~/minimal-agent.toml[/cyan] is created from the bundled default.
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
    built_tools = _build_tools(tools_opt, agent_cfg.root, searxng_url, raw)

    if max_turns is None:
        max_turns = int(get_config("ui.ma.max_turns", raw, 20))

    try:
        agent = Agent.from_config(agent_cfg, hooks=MaxTurnsHook(max_turns), tools=built_tools)
    except (KeyError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if prompt is not None:
        try:
            asyncio.run(_once(agent, prompt, json_out=json_out))
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc)
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
            asyncio.run(_repl(agent, agent_cfg, profile, tools_opt, searxng_url, max_turns, raw))
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc)
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed: {exc}", err=True)
            raise typer.Exit(1)


async def _once(agent: Agent, prompt: str, *, json_out: bool = False) -> None:
    messages = [Message(role="user", content=prompt)]
    if json_out:
        typer.echo(messages[0].model_dump_json(exclude_none=True))
        async for msg in agent.run(messages):
            if not msg.partial:
                typer.echo(msg.model_dump_json(exclude_none=True))
    else:
        final: str | None = None
        async for msg in agent.run(messages):
            if msg.role == "assistant" and not msg.partial and msg.content:
                final = msg.content
        if final is not None:
            typer.echo(final)


async def _repl(
    agent: Agent,
    agent_cfg: AgentConfig,
    profile: str,
    tools_opt: str | None,
    searxng_url: str | None,
    max_turns: int,
    raw_profile: dict | None = None,
) -> None:
    cwd_display = Path(agent_cfg.root or os.getcwd()).resolve()
    header = f"ma  profile={profile}  model={agent_cfg.model}"
    if tools_opt:
        header += f"  tools={tools_opt}  root={cwd_display}"
    _console.print(header, style="cyan")
    cmds = "/quit  /new  /model <name>"
    if tools_opt:
        cmds += "  /root <path>"
    _console.print(f"Commands: {cmds}\n", style="bright_black")

    state: dict = {"agent": agent}

    while True:
        try:
            user_input = typer.prompt("You")
        except EOFError, KeyboardInterrupt:
            break

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/new":
            state["agent"].reset()
            _console.print("Started new conversation.", style="green")
            continue
        if cmd.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model:
                state["agent"].model = new_model
                _console.print(f"Model changed to {new_model}", style="green")
            continue
        if tools_opt and cmd.startswith("/root "):
            new_root = user_input[6:].strip()
            if new_root:
                new_tools = _build_tools(tools_opt, new_root, searxng_url, raw_profile)
                new_cfg = dataclasses.replace(agent_cfg, root=new_root)
                state["agent"] = Agent.from_config(
                    new_cfg, hooks=MaxTurnsHook(max_turns), tools=new_tools
                )
                _console.print(f"Root changed to {Path(new_root).resolve()}", style="green")
            continue
        if not user_input.strip():
            continue

        messages = list(state["agent"].messages) + [Message(role="user", content=user_input)]
        typer.echo()

        try:
            async for msg in state["agent"].run(messages):
                if msg.role == "assistant":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            _console.print(
                                f"  tool: {tc.function.name}({tc.function.arguments})",
                                style="yellow",
                            )
                    elif msg.content and not msg.partial:
                        _console.print(f"  {msg.content}")
                elif msg.role == "tool":
                    display = (msg.content or "")[:_TOOL_RESULT_PREVIEW]
                    _console.print(f"  [{msg.name}] {display}", style="bright_black")
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc)
            state["agent"].reset()
        except httpx.TimeoutException:
            _console.print(
                f"Error: {agent_cfg.model} did not respond within {agent_cfg.llm_timeout}s"
                " (increase with --llm-timeout)",
                style="red",
            )
            state["agent"].reset()
        except httpx.RequestError as exc:
            _console.print(f"Error: request failed: {exc}", style="red")
            state["agent"].reset()
        typer.echo()
