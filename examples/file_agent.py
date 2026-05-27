#!/usr/bin/env python3
"""file-agent: a CLI that wraps the agent with built-in file tools.

Usage
-----
    # Interactive REPL (default)
    uv run python examples/file_agent.py --model gpt-4o-mini

    # One-shot / CI mode
    uv run python examples/file_agent.py --once "read pyproject.toml and summarize"

    # Restrict to a specific directory
    uv run python examples/file_agent.py --root /path/to/project
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer

from minimal_agent import Agent, Message, make_tools


def _build_agent(
    model: str,
    provider: str,
    system: str | None,
    root: str | None,
    timeout: float,
    max_turns: int,
    max_messages: int = 0,
    extra_body: dict | None = None,
) -> Agent:
    tools = make_tools(allowed_root=root)
    return Agent(
        model=model,
        provider=provider,
        system=system,
        tools=tools,
        timeout=timeout,
        max_turns=max_turns,
        max_messages=max_messages,
        extra_body=extra_body,
    )


app = typer.Typer(
    name="file-agent",
    help="Agent with built-in read / ls / edit tools",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _main() -> None:
    ...


@app.command()
def repl(
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider name"),
    system: str = typer.Option(
        None, "--system", "-s", help="System prompt (optional)",
    ),
    root: str = typer.Option(
        None, "--root", "-r",
        help="Allowed root directory (default: current working directory)",
    ),
    timeout: float = typer.Option(
        60.0, "--timeout", "-t", help="Timeout per LLM/tool call (seconds)",
    ),
    max_turns: int = typer.Option(
        20, "--max-turns", "-n", help="Maximum agent loop iterations",
    ),
    max_messages: int = typer.Option(
        0, "--max-messages", "-M",
        help="Stop after N yielded messages (0 = unlimited)",
    ),
    extra: str | None = typer.Option(
        None, "--extra", "-e",
        help="Extra JSON body params merged into the API request",
    ),
) -> None:
    """Start an interactive REPL session with file tools."""
    extra_body: dict | None = json.loads(extra) if extra else None

    cwd_display = Path(root or os.getcwd()).resolve()
    typer.echo(f"🤖 file-agent — model={model}  root={cwd_display}")
    typer.echo("Commands: /quit  /stop  /reset  /root <path>\n")

    async def _repl() -> None:
        # Keep mutable state
        state = {
            "agent": _build_agent(model, provider, system, root, timeout, max_turns, max_messages, extra_body),
            "messages": [],
            "root": root,
        }

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
                typer.echo("⏹  Agent stopped.")
                continue
            if cmd == "/reset":
                state["messages"].clear()
                typer.echo("🔄 Conversation reset.")
                continue
            if cmd.startswith("/root "):
                new_root = user_input[6:].strip()
                if new_root:
                    state["root"] = new_root
                    state["agent"] = _build_agent(
                        model, provider, system, new_root, timeout, max_turns,
                        max_messages, extra_body,
                    )
                    typer.echo(f"📁 Root changed to {Path(new_root).resolve()}")
                continue
            if not user_input.strip():
                continue

            state["messages"].append(Message(role="user", content=user_input))
            typer.echo()

            async for msg in state["agent"].run(state["messages"]):
                if msg.role == "assistant":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            typer.echo(
                                f"  🛠  {tc.function.name}({tc.function.arguments})"
                            )
                    elif msg.content:
                        typer.echo(f"  🤖 {msg.content}")
                elif msg.role == "tool":
                    display = (
                        msg.content[:300] + "…"
                        if msg.content and len(msg.content) > 300
                        else msg.content or ""
                    )
                    typer.echo(f"  📄 {msg.name}: {display}")
            typer.echo()

    asyncio.run(_repl())


@app.command()
def once(
    prompt: str = typer.Argument(..., help="User prompt"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name"),
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider name"),
    system: str = typer.Option(
        None, "--system", "-s", help="System prompt (optional)",
    ),
    root: str = typer.Option(
        None, "--root", "-r",
        help="Allowed root directory (default: current working directory)",
    ),
    timeout: float = typer.Option(
        60.0, "--timeout", "-t", help="Timeout per LLM/tool call (seconds)",
    ),
    max_turns: int = typer.Option(
        20, "--max-turns", "-n", help="Maximum agent loop iterations",
    ),
    max_messages: int = typer.Option(
        0, "--max-messages", "-M",
        help="Stop after N yielded messages (0 = unlimited)",
    ),
    extra: str | None = typer.Option(
        None, "--extra", "-e",
        help="Extra JSON body params merged into the API request",
    ),
) -> None:
    """Run a single prompt and exit (CI-friendly)."""
    extra_body: dict | None = json.loads(extra) if extra else None

    agent = _build_agent(model, provider, system, root, timeout, max_turns, max_messages, extra_body)

    async def _once() -> None:
        messages = [Message(role="user", content=prompt)]
        async for msg in agent.run(messages):
            if msg.role == "assistant":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        typer.echo(f"🛠  {tc.function.name}({tc.function.arguments})")
                elif msg.content:
                    typer.echo(msg.content)
            elif msg.role == "tool":
                typer.echo(f"📄 {msg.name}: {msg.content}", err=True)

    asyncio.run(_once())


if __name__ == "__main__":
    app()
