"""Execution modes: one-shot (-p PROMPT) and interactive REPL."""

import dataclasses
import os
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console

from ..agent import Agent
from ..config import AgentConfig
from ..hooks import MaxTurnsHook
from ..types import Message
from ._tools import build_tools

_console = Console()
_TOOL_RESULT_PREVIEW = 300
_HTTP_ERROR_BODY_PREVIEW = 500

_HTTP_HINTS: dict[int, str] = {
    401: "Authentication failed. Check your API key.",
    403: "Access denied. Your API key may lack permissions for this resource.",
    404: "Endpoint not found. Check the base URL and model name.",
    429: "Rate limited. Wait before retrying, or reduce request frequency.",
}


def print_http_error(exc: httpx.HTTPStatusError) -> None:
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


async def once(agent: Agent, prompt: str, *, json_out: bool = False) -> None:
    """Run the agent once and print the final response."""
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


async def repl(
    agent: Agent,
    agent_cfg: AgentConfig,
    profile: str,
    tools_opt: str | None,
    searxng_url: str | None,
    max_turns: int,
    raw_profile: dict[str, Any] | None = None,
) -> None:
    """Run the interactive REPL until the user quits."""
    cwd_display = Path(agent_cfg.root or os.getcwd()).resolve()
    header = f"profile={profile}  model={agent_cfg.model}"
    if tools_opt:
        header += f"  tools={tools_opt}  root={cwd_display}"
    _console.print(header, style="cyan")
    cmds = "/quit  /new  /model <name>"
    if tools_opt:
        cmds += "  /root <path>"
    _console.print(f"Commands: {cmds}\n", style="bright_black")

    current_agent = agent

    while True:
        try:
            user_input = typer.prompt("You")
        except EOFError, KeyboardInterrupt:
            break

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/new":
            current_agent.reset()
            _console.print("Started new conversation.", style="green")
            continue
        if cmd.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model:
                current_agent.model = new_model
                _console.print(f"Model changed to {new_model}", style="green")
            continue
        if tools_opt and cmd.startswith("/root "):
            new_root = user_input[6:].strip()
            if new_root:
                new_cfg = dataclasses.replace(agent_cfg, root=new_root)
                current_agent = Agent.from_config(
                    new_cfg,
                    hooks=MaxTurnsHook(max_turns),
                    tools=build_tools(tools_opt, new_root, searxng_url, raw_profile),
                )
                _console.print(f"Root changed to {Path(new_root).resolve()}", style="green")
            continue
        if not user_input.strip():
            continue

        messages = list(current_agent.messages) + [Message(role="user", content=user_input)]
        typer.echo()

        try:
            async for msg in current_agent.run(messages):
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
            print_http_error(exc)
            current_agent.reset()
        except httpx.TimeoutException:
            _console.print(
                f"Error: {agent_cfg.model} did not respond within {agent_cfg.llm_timeout}s"
                " (increase with --llm-timeout)",
                style="red",
            )
            current_agent.reset()
        except httpx.RequestError as exc:
            _console.print(f"Error: request failed: {exc}", style="red")
            current_agent.reset()
        typer.echo()
