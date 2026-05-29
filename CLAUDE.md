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
uv run ma --tools read,edit,ls,search --root .

# Run with thinking enabled (reasoning_effort=medium)
uv run ma -p "your prompt here" --think
```

## Architecture

The library is a minimal, dependency-light agent loop built on top of any OpenAI-compatible API. The main layers:

**`types.py`** — Pydantic models: `Message`, `ToolCall`, `FunctionCall`, `Usage`. These mirror the OpenAI chat format closely. `Message.reasoning` is a streaming-only field (for thinking models like DeepSeek/Qwen3) that is deliberately excluded when messages are sent back to the API.

**`_config.py`** — Reads `.ma-config.toml` (env var > `~/.ma-config.toml` > package-local > repo root) to resolve defaults (`[defaults]`), per-model overrides (`[models.<name>]`), and providers (`[providers.<name>]`). Providers: `openai`, `ollama`, `together`, `groq`, `anthropic`, `openrouter`, `deepseek`.

**`_client.py`** — `stream_chat()`: a raw `httpx`-based SSE streaming client that yields progressively built `Message` objects. Handles both text content and incremental tool-call assembly (stitching together streamed `tool_calls` deltas by index).

**`tool.py`** — `@tool` decorator that registers async functions into a global `_TOOL_REGISTRY`. Infers JSON Schema parameters from Python type annotations automatically. Tools registered this way are globally visible to all `Agent` instances. Defines the `ToolInfo` dataclass.

**`tools/`** — `make_tools(allowed_root)` factory (in `tools/__init__.py`) returning scoped `read`, `ls`, `edit`, and `search` tools (one module each). These are constructed with closures (not `@tool`) so the allowed root directory is captured per-call rather than globally, and path traversal is enforced. Pass the returned list as `tools=` to `Agent`. `web_tools.py` similarly provides the optional `make_web_tools` (`web_search`, `web_fetch`) behind the `[web]` extra.

**`_execution.py`** — `exec_tool()`: parses tool-call arguments, runs `on_before_tool`/`on_after_tool` hooks, validates arguments against the tool's JSON Schema, and executes the tool with a timeout, formatting any error into the returned result string.

**`agent.py`** — `Agent` class: the main loop. Each `run()` call sends the message history → LLM → executes tool calls → loops until the LLM responds without tool calls. Yields `Message` objects as they arrive (streaming content and tool results). The core has no built-in turn limit — bound a run with a hook that calls `agent.stop()` (e.g. `MaxTurnsHook`). `agent.stop()` requests a clean exit after the current turn; `agent.abort()` cancels immediately. After `run()`, `agent.messages` holds the full history (system prompt, assistant turns, tool results) for reuse or handoff.

**`hooks.py`** — `AgentHooks` base class: lifecycle callbacks (`on_before_agent`, `on_before_turn`, `on_before_llm`, `on_after_llm`, `on_before_tool`, `on_after_tool`, `on_after_turn`, `on_after_agent`). All are no-op by default and called via `_safe_hook`, which logs and swallows exceptions so a buggy hook cannot crash the loop.

**`_cache.py`** — Disk cache (via `diskcache`) for LLM responses, keyed by a SHA-256 of the request. `Agent` enables it by default (`cache_dir`); pass `cache_dir=None` to disable.

**`_cli.py`** — Typer CLI (`ma`). No subcommands: bare `ma` starts an interactive REPL; `ma -p PROMPT` runs once and prints the final response. `--tools read,edit,ls,search` enables built-in file tools (scoped to `--root`); `--max-turns` caps loop iterations via `MaxTurnsHook`. Supports `--think` / `--extra` for provider-specific `extra_body` fields.

**`__init__.py`** — Public API exports: `Agent`, `AgentHooks`, `Message`, `ToolCall`, `FunctionCall`, `Usage`, `ToolInfo`, `tool`, `clear_registry`, `make_tools`, `make_web_tools`.

### Tool registration

Two ways to provide tools to `Agent`:

1. **Global registry** via `@tool` decorator — auto-discovered by all `Agent` instances.
2. **Per-agent tools** via `tools=[...]` constructor arg — useful for scoped tools like `make_tools()` where state (root dir) must be isolated.

When both exist, per-agent tools take name-priority over global ones.

### Provider configuration

`.ma-config.toml` uses three top-level sections: ``[defaults]`` (provider, model, tools), ``[models.<name>]`` (per-model overrides merged on top of defaults), and ``[providers.<name>]`` (base_url, env_key_name). Read by ``_config.py`` with priority ``$MA_CONFIG_PATH`` > ``~/.ma-config.toml`` > package-local > repo root. The `ollama` provider has no `env_key_name` (no auth required). All others read their key from the environment variable named in `env_key_name`.
