"""Typer-based CLI for ``ma``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer

from ._agent import Agent
from ._builtin_tools import make_tools
from ._tool import ToolInfo
from ._web_tools import make_web_tools
from ._types import Message

app = typer.Typer(
    name="ma",
    help="minimal-agent – a minimal LLM agent with tool support",
    invoke_without_command=True,
)

_BUILTIN_TOOL_NAMES = {"read", "ls", "edit"}
_WEB_TOOL_NAMES = {"web_search", "web_fetch"}
_ALL_TOOL_NAMES = _BUILTIN_TOOL_NAMES | _WEB_TOOL_NAMES


def _version_callback(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"minimal-agent {__version__}")
        raise typer.Exit()


def _build_tools(
    tools_opt: str | None, root: str | None, searxng_url: str | None = None
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
        fs_tools = make_tools(allowed_root=root)
        result.extend(t for t in fs_tools if t.name in names)
    if names & _WEB_TOOL_NAMES:
        url = searxng_url or os.environ.get("SEARXNG_URL")
        web_tools = make_web_tools(searxng_url=url)
        result.extend(t for t in web_tools if t.name in names)
    return result


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Run once and print final result (non-interactive)"
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name"),
    provider: str = typer.Option("openai", "--provider", help="Provider name"),
    system: Optional[str] = typer.Option(
        None, "--system", "-s", help="System prompt (optional)"
    ),
    tools_opt: Optional[str] = typer.Option(
        None, "--tools",
        help="Comma-separated built-in tools to enable: read,edit,ls",
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
    timeout: float = typer.Option(
        60.0, "--timeout", "-t", help="Timeout for LLM and tool calls"
    ),
    max_turns: int = typer.Option(
        20, "--max-turns", "-n", help="Maximum agent loop iterations"
    ),
    max_messages: int = typer.Option(
        0, "--max-messages", "-M",
        help="Stop after N yielded messages (0 = unlimited)",
    ),
    think: bool = typer.Option(
        False, "--think/--no-think",
        help="Enable (medium) or disable (none) reasoning_effort",
    ),
    extra: Optional[str] = typer.Option(
        None, "--extra", "-e",
        help="Extra JSON body params merged into the API request",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Output all non-partial messages as JSONL (one JSON object per line)"
    ),
    _version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Start an interactive REPL, or run once with -p PROMPT."""
    if ctx.invoked_subcommand is not None:
        return

    extra_body: dict | None = json.loads(extra) if extra else None
    extra_body = (extra_body or {}) | {"reasoning_effort": "medium" if think else "none"}

    tools = _build_tools(tools_opt, root, searxng_url)

    agent = Agent(
        model=model,
        provider=provider,
        system=system,
        tools=tools,
        timeout=timeout,
        max_turns=max_turns,
        max_messages=max_messages,
        extra_body=extra_body,
    )

    if prompt is not None:
        asyncio.run(_once(agent, prompt, json_out=json_out))
    else:
        asyncio.run(_repl(agent, model, provider, root, tools_opt, searxng_url,
                          system, timeout, max_turns, max_messages, extra_body))


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
) -> None:
    cwd_display = Path(root or os.getcwd()).resolve()
    header = f"ma — model={model} provider={provider}"
    if tools_opt:
        header += f"  tools={tools_opt}  root={cwd_display}"
    typer.echo(header)
    cmds = "/quit  /stop  /reset"
    if tools_opt:
        cmds += "  /root <path>"
    typer.echo(f"Commands: {cmds}\n")

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
            typer.echo("Agent stopped.")
            continue
        if cmd == "/reset":
            state["agent"].reset()
            typer.echo("Conversation reset.")
            continue
        if tools_opt and cmd.startswith("/root "):
            new_root = user_input[6:].strip()
            if new_root:
                state["root"] = new_root
                new_tools = _build_tools(tools_opt, new_root, searxng_url)
                state["agent"] = Agent(
                    model=model,
                    provider=provider,
                    system=system,
                    tools=new_tools,
                    timeout=timeout,
                    max_turns=max_turns,
                    max_messages=max_messages,
                    extra_body=extra_body,
                )
                typer.echo(f"Root changed to {Path(new_root).resolve()}")
            continue
        if not user_input.strip():
            continue

        messages = list(state["agent"].conversation) + [Message(role="user", content=user_input)]
        typer.echo()

        async for msg in state["agent"].run(messages):
            if msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        typer.echo(f"  tool: {tc.function.name}({tc.function.arguments})")
                elif msg.content and not msg.partial:
                    typer.echo(f"  {msg.content}")
            elif msg.role == "tool":
                display = (msg.content or "")[:300]
                typer.echo(f"  [{msg.name}] {display}")
        typer.echo()
