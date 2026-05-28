# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_integration.py

# Run a single test by name
uv run pytest tests/test_integration.py::test_stream_chat_text

# Lint
uv run ruff check src tests

# Format
uv run ruff format src tests

# Run the CLI (interactive REPL)
uv run ma --model gpt-4o-mini --provider openai

# Run the CLI (one-shot, print final result)
uv run ma -p "your prompt here" --model gpt-4o-mini

# Run with file tools enabled
uv run ma --tools read,edit,ls --root .

# Run with thinking enabled (reasoning_effort=medium)
uv run ma -p "your prompt here" --think
```

## Architecture

The library is a minimal, dependency-light agent loop built on top of any OpenAI-compatible API. The main layers:

**`_types.py`** — Pydantic models: `Message`, `ToolCall`, `FunctionCall`, `Usage`. These mirror the OpenAI chat format closely. `Message.reasoning` is a streaming-only field (for thinking models like DeepSeek/Qwen3) that is deliberately excluded when messages are sent back to the API.

**`_provider.py`** — Reads `.ma-config.toml` (env var > `~/.ma-config.toml` > package-local > repo root) to resolve defaults (`[defaults]`) and providers (`[providers.<name>]`). Providers: `openai`, `ollama`, `together`, `groq`, `anthropic`, `openrouter`, `deepseek`.

**`_client.py`** — `stream_chat()`: a raw `httpx`-based SSE streaming client that yields progressively built `Message` objects. Handles both text content and incremental tool-call assembly (stitching together streamed `tool_calls` deltas by index).

**`_tool.py`** — `@tool` decorator that registers async functions into a global `_TOOL_REGISTRY`. Infers JSON Schema parameters from Python type annotations automatically. Tools registered this way are globally visible to all `Agent` instances.

**`_builtin_tools.py`** — `make_tools(allowed_root)` factory that returns scoped `read`, `ls`, and `edit` tools. These are constructed with closures (not `@tool`) so the allowed root directory is captured per-call rather than globally, and path traversal is enforced. Pass the returned list as `tools=` to `Agent`.

**`_agent.py`** — `Agent` class: the main loop. Each `run()` call sends conversation → LLM → executes tool calls → loops until the LLM responds without tool calls. Yields `Message` objects as they arrive (streaming content and tool results). Tool execution is async with timeout. `agent.stop()` cancels the current run and sets a stop event checked at each iteration. After `run()` completes, `agent.conversation` holds the full history (including system prompt, assistant turns, and tool results) for reuse or handoff.

**`_cli.py`** — Typer CLI (`ma`). No subcommands: bare `ma` starts an interactive REPL; `ma -p PROMPT` runs once and prints the final response. `--tools read,edit,ls` enables built-in file tools (scoped to `--root`). Supports `--think` / `--extra` for provider-specific `extra_body` fields.

**`__init__.py`** — Public API exports: `Agent`, `Message`, `tool`, `make_tools`, `ToolInfo`.

### Tool registration

Two ways to provide tools to `Agent`:

1. **Global registry** via `@tool` decorator — auto-discovered by all `Agent` instances.
2. **Per-agent tools** via `tools=[...]` constructor arg — useful for scoped tools like `make_tools()` where state (root dir) must be isolated.

When both exist, per-agent tools take name-priority over global ones.

### Provider configuration

`.ma-config.toml` uses two top-level sections: ``[defaults]`` (provider, model, tools) and ``[providers.<name>]`` (base_url, env_key_name). Read by ``_provider.py`` with priority ``$MA_CONFIG_PATH`` > ``~/.ma-config.toml`` > package-local > repo root. The `ollama` provider has no `env_key_name` (no auth required). All others read their key from the environment variable named in `env_key_name`.
