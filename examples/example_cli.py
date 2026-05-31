"""Full-featured CLI for minimal-agent.

This is a reference implementation showing how to build a richer CLI on top
of the minimal-agent library. It adds:
  - Config file (~/.ma-config.toml) with named profiles
  - Built-in file tools (read, ls, edit, grep) and web tools
  - REPL commands: /new, /model, /root

To use it as your 'ma' command, update pyproject.toml:
    [project.scripts]
    ma = "examples.example_cli:app"

Or run directly:
    uv run python examples/example_cli.py --profile openai -p "hello"
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
import typer
from rich.console import Console

from minimal_agent import Agent, AgentConfig, MaxTurnsHook, Message
from minimal_agent.profiles import _load_merged_profile, config_path, get_config, resolve_profile
from minimal_agent._logging import setup_logging
from minimal_agent.registry import ToolInfo
from minimal_agent.tools import make_tools
from minimal_agent.web_tools import make_web_tools

_console = Console()
_HTTP_ERROR_BODY_PREVIEW = 500
_TOOL_RESULT_PREVIEW = 300

_BUILTIN_TOOL_NAMES = {"read", "ls", "edit", "grep"}
_WEB_TOOL_NAMES = {"web_search", "web_fetch"}
_ALL_TOOL_NAMES = _BUILTIN_TOOL_NAMES | _WEB_TOOL_NAMES

_HTTP_HINTS: dict[int, str] = {
    401: "Authentication failed. Check your API key.",
    403: "Access denied. Your API key may lack permissions for this resource.",
    404: "Endpoint not found. Check the provider base_url and model name.",
    429: "Rate limited. Wait before retrying, or reduce request frequency.",
}

try:
    _DEFAULTS: dict = _load_merged_profile("default")
except FileNotFoundError:
    _DEFAULTS = {}

app = typer.Typer(
    name="ma",
    help="ma: a minimal LLM agent with tools and config",
    invoke_without_command=True,
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
        from minimal_agent import __version__
        typer.echo(f"minimal-agent {__version__}")
        raise typer.Exit()


def _ensure_home_config() -> None:
    dst = Path.home() / ".ma-config.toml"
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
    if names & _BUILTIN_TOOL_NAMES:
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
    if names & _WEB_TOOL_NAMES:
        url = searxng_url or os.environ.get("SEARXNG_URL")
        try:
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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    profile: str = typer.Option("default", "--profile", help="Named profile from .ma-config.toml"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name (overrides profile)"),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Run once and print result (non-interactive)",
    ),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
    think: bool | None = typer.Option(
        None, "--think/--no-think",
        help="Enable (medium) or disable (none) reasoning_effort",
    ),
    extra: str | None = typer.Option(
        None, "--extra", "-e", help="Extra JSON body params merged into the API request",
    ),
    tools_opt: str | None = typer.Option(
        None, "--tools",
        help="Comma-separated tools to enable: read,edit,ls,grep",
        metavar="TOOLS",
    ),
    root: str | None = typer.Option(
        None, "--root", "-r", help="Allowed root directory for file tools (default: cwd)",
    ),
    searxng_url: str | None = typer.Option(
        None, "--searxng-url",
        help="SearXNG base URL for web tools (overrides SEARXNG_URL env var)",
        envvar="SEARXNG_URL",
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", "-n", help="Maximum agent loop iterations"),
    timeout: float | None = typer.Option(None, "--timeout", "-t", help="LLM response timeout in seconds"),
    json_out: bool = typer.Option(False, "--json", help="Output all non-partial messages as JSONL"),
    log_level: str | None = typer.Option(
        None, "--log-level", "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
        metavar="LEVEL",
    ),
    _version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Start an interactive REPL, or run once with -p PROMPT."""
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
    if tools_opt is None and agent_cfg.tools:
        tools_opt = ",".join(agent_cfg.tools)
    if system is not None:
        agent_cfg.system = system
    if max_turns is None:
        max_turns = int(get_config("ui.example_cli.max_turns", raw, 50))
    if searxng_url is None:
        searxng_url = get_config("tool.web_search.url", raw) or os.environ.get("SEARXNG_URL")
    if timeout is not None:
        agent_cfg.llm_timeout = timeout

    if extra:
        try:
            agent_cfg.llm_extra_body = json.loads(extra)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: --extra must be valid JSON: {exc}", err=True)
            raise typer.Exit(1)

    if think is not None:
        agent_cfg.llm_extra_body = (agent_cfg.llm_extra_body or {}) | {
            "reasoning_effort": "medium" if think else "none"
        }

    if root:
        agent_cfg.root = root
    built_tools = _build_tools(tools_opt, root, searxng_url, raw)

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
                f"Error: {profile}/{agent_cfg.model} did not respond within {agent_cfg.llm_timeout}s"
                " (increase with --timeout)",
                err=True,
            )
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed: {exc}", err=True)
            raise typer.Exit(1)
    else:
        try:
            asyncio.run(_repl(agent, agent_cfg, profile, root, tools_opt,
                              searxng_url, max_turns, raw))
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
    root: str | None,
    tools_opt: str | None,
    searxng_url: str | None,
    max_turns: int,
    raw_profile: dict | None = None,
) -> None:
    import dataclasses

    cwd_display = Path(root or os.getcwd()).resolve()
    header = f"ma  profile={profile}  model={agent_cfg.model}"
    if tools_opt:
        header += f"  tools={tools_opt}  root={cwd_display}"
    _console.print(header, style="cyan")
    cmds = "/quit  /new  /model <name>"
    if tools_opt:
        cmds += "  /root <path>"
    _console.print(f"Commands: {cmds}\n", style="bright_black")

    state: dict = {"agent": agent, "root": root}

    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
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
                state["root"] = new_root
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
                f"Error: {profile}/{agent_cfg.model} did not respond within {agent_cfg.llm_timeout}s"
                " (increase with --timeout)",
                style="red",
            )
            state["agent"].reset()
        except httpx.RequestError as exc:
            _console.print(f"Error: request failed: {exc}", style="red")
            state["agent"].reset()
        typer.echo()


if __name__ == "__main__":
    app()
