"""ma: minimal LLM agent CLI — no config file, no tools, no hooks."""

import asyncio
import json

import httpx
import typer
from rich.console import Console

from . import __version__
from ._logging import setup_logging
from .agent import Agent
from .config import AgentConfig
from .hooks import MaxTurnsHook
from .types import Message

_console = Console()
_HTTP_ERROR_BODY_PREVIEW = 500

app = typer.Typer(name="ma", help="ma: a minimal LLM agent", invoke_without_command=True)

_HTTP_HINTS: dict[int, str] = {
    401: "Authentication failed. Check your API key.",
    403: "Access denied. Your API key may lack permissions for this resource.",
    404: "Endpoint not found. Check the base URL and model name.",
    429: "Rate limited. Wait before retrying, or reduce request frequency.",
}


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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Model name", envvar="MA_MODEL"),
    base_url: str = typer.Option(
        "http://localhost:11434/v1", "--base-url",
        help="API base URL", envvar="MA_BASE_URL",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key", envvar="MA_API_KEY",
    ),
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
    max_turns: int = typer.Option(20, "--max-turns", "-n", help="Maximum agent loop iterations"),
    timeout: float = typer.Option(300.0, "--timeout", "-t", help="LLM response timeout in seconds"),
    json_out: bool = typer.Option(False, "--json", help="Output messages as JSONL"),
    log_level: str | None = typer.Option(
        None, "--log-level", "-l", help="Logging level: DEBUG, INFO, WARNING, ERROR",
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

    if extra:
        try:
            llm_extra_body: dict | None = json.loads(extra)
        except json.JSONDecodeError as exc:
            typer.echo(f"Error: --extra must be valid JSON: {exc}", err=True)
            raise typer.Exit(1)
    else:
        llm_extra_body = None

    if think is not None:
        llm_extra_body = (llm_extra_body or {}) | {"reasoning_effort": "medium" if think else "none"}

    cfg = AgentConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system=system,
        llm_timeout=timeout,
        llm_extra_body=llm_extra_body,
    )
    agent = Agent.from_config(cfg, hooks=MaxTurnsHook(max_turns))

    if prompt is not None:
        try:
            asyncio.run(_once(agent, prompt, json_out=json_out))
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc)
            raise typer.Exit(1)
        except httpx.TimeoutException:
            typer.echo(
                f"Error: {model} did not respond within {timeout}s (increase with --timeout)",
                err=True,
            )
            raise typer.Exit(1)
        except httpx.RequestError as exc:
            typer.echo(f"Error: request failed: {exc}", err=True)
            raise typer.Exit(1)
    else:
        asyncio.run(_repl(agent, model, base_url, timeout))


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


async def _repl(agent: Agent, model: str, base_url: str, timeout: float) -> None:
    _console.print(f"ma  model={model}  base-url={base_url}", style="cyan")
    _console.print("Commands: /quit  /new  /model <name>\n", style="bright_black")

    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/new":
            agent.reset()
            _console.print("Started new conversation.", style="green")
            continue
        if cmd.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model:
                agent.model = new_model
                _console.print(f"Model changed to {new_model}", style="green")
            continue
        if not user_input.strip():
            continue

        messages = list(agent.messages) + [Message(role="user", content=user_input)]
        typer.echo()

        try:
            async for msg in agent.run(messages):
                if msg.role == "assistant" and msg.content and not msg.partial:
                    _console.print(f"  {msg.content}")
        except httpx.HTTPStatusError as exc:
            _print_http_error(exc)
            agent.reset()
        except httpx.TimeoutException:
            _console.print(
                f"Error: {agent.model} did not respond within {timeout}s"
                " (increase with --timeout)",
                style="red",
            )
            agent.reset()
        except httpx.RequestError as exc:
            _console.print(f"Error: request failed: {exc}", style="red")
            agent.reset()
        typer.echo()
