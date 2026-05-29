"""Typer-based CLI for ``ma``."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console

from .agent import Agent
from .hooks import AgentHooks
from ._builtin_tools import make_tools
from ._config import DefaultConfig, config_path, resolve_defaults, resolve_model_config
from ._logging import setup_logging
from .tool import ToolInfo
from ._web_tools import make_web_tools
from .types import Message

_DEFAULTS = resolve_defaults()
_console = Console()
_HTTP_ERROR_BODY_PREVIEW = 500
_TOOL_RESULT_PREVIEW = 300


class _MaxTurnsHook(AgentHooks):
    def __init__(self, n: int) -> None:
        self._n = n
        self._turns = 0

    async def on_after_turn(self, agent: Agent) -> None:
        self._turns += 1
        if self._turns >= self._n:
            agent.stop()

app = typer.Typer(
    name="ma",
    help="minimal-agent – a minimal LLM agent with tool support",
    invoke_without_command=True,
)

_BUILTIN_TOOL_NAMES = {"read", "ls", "edit", "search"}
_WEB_TOOL_NAMES = {"web_search", "web_fetch"}
_ALL_TOOL_NAMES = _BUILTIN_TOOL_NAMES | _WEB_TOOL_NAMES


def _version_callback(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"minimal-agent {__version__}")
        raise typer.Exit()


def _ensure_home_config() -> None:
    """Auto-install ~/.ma-config.toml on first run if it doesn't exist."""
    dst = Path.home() / ".ma-config.toml"
    if dst.exists():
        return
    src = config_path()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Originally installed by minimal-agent from {src}\n"
        "# Edit this file to configure provider, model, and default tools.\n\n"
    )
    dst.write_text(header + src.read_text())


def _build_tools(
    tools_opt: str | None,
    root: str | None,
    searxng_url: str | None = None,
    cfg: DefaultConfig | None = None,
) -> list[ToolInfo] | None:
    """Return a filtered list of tools, or None if no tools requested."""
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
            read_max_lines=cfg.tool_read_max_lines if cfg else 100,
            search_max_chars=cfg.tool_search_max_chars if cfg else 20_000,
            search_timeout=cfg.tool_search_timeout if cfg else 30.0,
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
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Run once and print final result (non-interactive)"
    ),
    model: str = typer.Option(
        _DEFAULTS.model, "--model", "-m", help="Model name"
    ),
    provider: str = typer.Option(
        _DEFAULTS.provider, "--provider", help="Provider name"
    ),
    system: Optional[str] = typer.Option(
        None, "--system", "-s", help="System prompt (optional)"
    ),
    tools_opt: Optional[str] = typer.Option(
        None, "--tools",
        help="Comma-separated built-in tools to enable: read,edit,ls,search",
        metavar="TOOLS",
    ),
    root: Optional[str] = typer.Option(
        None, "--root", "-r",
        help="Allowed root directory for file tools (default: cwd)",
    ),
    searxng_url: Optional[str] = typer.Option(
        None, "--searxng-url",
        help="SearXNG base URL for web tools (overrides SEARXNG_URL env var)",
        envvar="SEARXNG_URL",
    ),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", "-t", help="Timeout for LLM and tool calls"
    ),
    max_turns: Optional[int] = typer.Option(
        None, "--max-turns", "-n", help="Maximum agent loop iterations"
    ),
    max_messages: Optional[int] = typer.Option(
        None, "--max-messages", "-M",
        help="Stop after N yielded messages (0 = unlimited)",
    ),
    think: Optional[bool] = typer.Option(
        None, "--think/--no-think",
        help="Enable (medium) or disable (none) reasoning_effort",
    ),
    extra: Optional[str] = typer.Option(
        None, "--extra", "-e",
        help="Extra JSON body params merged into the API request",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Output all non-partial messages as JSONL (one JSON object per line)"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: no logging)",
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
        setup_logging(log_level.upper())

    _ensure_home_config()

    # Resolve model-specific overrides from config
    cfg = resolve_model_config(model)

    # Apply model config as fallbacks for unset CLI options
    if tools_opt is None and cfg.tools:
        tools_opt = ",".join(cfg.tools)
    if system is None and cfg.system:
        system = cfg.system
    if think is None:
        think = cfg.think
    if extra is None and cfg.extra:
        extra = json.dumps(cfg.extra)
    if max_turns is None:
        max_turns = cfg.max_turns
    if max_messages is None:
        max_messages = cfg.max_messages
    if searxng_url is None and cfg.searxng_url:
        searxng_url = cfg.searxng_url
    resolved_timeout: float = timeout if timeout is not None else cfg.llm_timeout

    extra_body: dict | None = json.loads(extra) if extra else None
    extra_body = (extra_body or {}) | {"reasoning_effort": "medium" if think else "none"}

    tools = _build_tools(tools_opt, root, searxng_url, cfg)

    try:
        agent = Agent(
            model=model,
            provider=provider,
            system=system,
            tools=tools,
            timeout=resolved_timeout,
            hooks=_MaxTurnsHook(max_turns),
            max_messages=max_messages,
            extra_body=extra_body,
        )
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if prompt is not None:
        try:
            asyncio.run(_once(agent, prompt, json_out=json_out))
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} from {exc.request.url}", err=True)
            body = exc.response.text[:_HTTP_ERROR_BODY_PREVIEW]
            if body:
                typer.echo(f"  {body}", err=True)
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed — {exc}", err=True)
            raise typer.Exit(1)
    else:
        try:
            asyncio.run(_repl(agent, model, provider, root, tools_opt, searxng_url,
                              system, resolved_timeout, max_turns, max_messages, extra_body, cfg))
        except httpx.HTTPStatusError as exc:
            typer.echo(f"Error: {exc.response.status_code} from {exc.request.url}", err=True)
            body = exc.response.text[:_HTTP_ERROR_BODY_PREVIEW]
            if body:
                typer.echo(f"  {body}", err=True)
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed — {exc}", err=True)
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
    model: str,
    provider: str,
    root: str | None,
    tools_opt: str | None,
    searxng_url: str | None,
    system: str | None,
    timeout: float,
    max_turns: int,
    max_messages: int,
    extra_body: dict | None,
    cfg: DefaultConfig | None = None,
) -> None:
    cwd_display = Path(root or os.getcwd()).resolve()
    header = f"ma — model={model} provider={provider}"
    if tools_opt:
        header += f"  tools={tools_opt}  root={cwd_display}"
    _console.print(header, style="cyan")
    cmds = "/quit  /stop  /new"
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
        if cmd == "/stop":
            state["agent"].stop()
            _console.print("Agent stopped.", style="yellow")
            continue
        if cmd == "/new":
            state["agent"].stop()
            state["agent"].reset()
            _console.print("Started new conversation.", style="green")
            continue
        if tools_opt and cmd.startswith("/root "):
            new_root = user_input[6:].strip()
            if new_root:
                state["root"] = new_root
                new_tools = _build_tools(tools_opt, new_root, searxng_url, cfg)
                state["agent"] = Agent(
                    model=model,
                    provider=provider,
                    system=system,
                    tools=new_tools,
                    timeout=timeout,
                    hooks=_MaxTurnsHook(max_turns),
                    max_messages=max_messages,
                    extra_body=extra_body,
                )
                _console.print(f"Root changed to {Path(new_root).resolve()}", style="green")
            continue
        if not user_input.strip():
            continue

        messages = list(state["agent"].conversation) + [Message(role="user", content=user_input)]
        typer.echo()

        async for msg in state["agent"].run(messages):
            if msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        _console.print(f"  tool: {tc.function.name}({tc.function.arguments})", style="yellow")
                elif msg.content and not msg.partial:
                    _console.print(f"  {msg.content}")
            elif msg.role == "tool":
                display = (msg.content or "")[:_TOOL_RESULT_PREVIEW]
                _console.print(f"  [{msg.name}] {display}", style="bright_black")
        typer.echo()

