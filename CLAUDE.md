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

# Run the CLI (interactive REPL) — model is required
uv run ma --model qwen3.5:9b

# Run the CLI with a custom base URL and API key
uv run ma --model gpt-4o-mini --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY

# Run the CLI (one-shot, print final result)
uv run ma -p "your prompt here" --model qwen3.5:9b

# Run with file tools enabled
uv run ma --tools read,edit,ls,grep --root . --model qwen3.5:9b

# Run with bash tool enabled
uv run ma --tools bash --root . --model qwen3.5:9b

# Run with thinking enabled (reasoning_effort=medium)
uv run ma -p "your prompt here" --think --model qwen3.5:9b

# Run with a named profile from .ma-config.toml
uv run ma --profile openai -p "your prompt here"
```

## Architecture

The library is a minimal, dependency-light agent loop built on top of any OpenAI-compatible API. The main layers:

**`types.py`** — Pydantic models: `Message`, `ToolCall`, `FunctionCall`, `Usage`. These mirror the OpenAI chat format closely. `Message.reasoning` is a streaming-only field (for thinking models like DeepSeek/Qwen3) that is deliberately excluded when messages are sent back to the API.

**`config.py`** — `AgentConfig` dataclass: a portable, serialisable bundle of Agent constructor parameters (`model`, `base_url`, `api_key`, `system`, `tools`, `root`, timeouts, `llm_extra_body`, `cache_dir`). Use `Agent.from_config(cfg)` or `dataclasses.replace(cfg, ...)` to derive variants. Hooks are excluded — they are stateful runtime objects.

**`profiles.py`** — Loads `.ma-config.toml` (env var > `~/.ma-config.toml` > package-local > repo root) and resolves named profiles. Config structure: `[profiles.default]` is the base; `[profiles.<name>]` merges on top; `[config]` is a global settings tree (deep-merged with per-profile `[profiles.<name>.config]`). `{{VAR_NAME}}` in any string value is resolved from the environment. Call `resolve_profile("name")` → `AgentConfig`. Call `get_config("tool.read.max_lines", raw)` to read settings from the `[config]` tree.

**`_api_client.py`** — `stream_chat()`: a raw `httpx`-based SSE streaming client that yields progressively built `Message` objects. Handles both text content and incremental tool-call assembly (stitching together streamed `tool_calls` deltas by index). Caches responses by SHA-256 request key when a cache is provided.

**`registry.py`** — `@tool` decorator that registers async functions into a global `_TOOL_REGISTRY`. Infers JSON Schema parameters from Python type annotations automatically. Defines the `ToolInfo` dataclass. Also exposes `get_tool(name)`, `list_tools()`, `clear_registry()`.

**`tools/`** — Built-in file tools (one module each): `read`, `ls`, `edit`, `grep`. `make_tools(allowed_root)` in `tools/__init__.py` returns all four as a list of `ToolInfo`. All are constructed with closures so the allowed root is captured per-call and path traversal is enforced. `make_grep_tool` wraps `rg` (ripgrep). `make_bash_tool` in `tools/bash.py` runs shell commands via `bash -c` with dangerous-pattern blocking, working-directory scoping, output truncation, and timeout/process-group kill. `web_tools.py` provides the optional `make_web_tools` (`web_search`, `web_fetch`) behind the `[web]` extra.

**`_tool_execution.py`** — `exec_tool()`: parses tool-call arguments, runs `on_before_tool`/`on_after_tool` hooks, validates arguments against the tool's JSON Schema, and executes the tool with a timeout, formatting any error into the returned result string.

**`agent.py`** — `Agent` class: the main loop. Accepts `tools` as a mixed list of tool name strings or `ToolInfo` objects — names are resolved to built-in file tools (scoped to `root`), web tools, bash tool, or globally registered `@tool` functions. Each `run()` call sends the message history → LLM → executes tool calls → loops until the LLM responds without tool calls. Yields `Message` objects as they arrive. `agent.stop()` requests a clean exit after the current turn; `agent.abort()` cancels immediately. `Agent.from_config(cfg, hooks=...)` is the preferred constructor.

**`hooks.py`** — `AgentHooks` base class: lifecycle callbacks (`on_before_agent`, `on_before_turn`, `on_before_llm`, `on_after_llm`, `on_before_tool`, `on_after_tool`, `on_after_turn`, `on_after_agent`). `MaxTurnsHook` is a built-in hook that calls `agent.stop()` after N turns. All hooks are no-op by default and called via `_safe_hook`.

**`_cache.py`** — Disk cache (via `diskcache`) for LLM responses, keyed by a SHA-256 of the request. `Agent` enables it by default (`cache_dir`); pass `cache_dir=None` to disable.

**`minimal_cli.py`** — Typer CLI (`ma`). `--model` is required (or set via `MA_MODEL` env var). `--base-url` defaults to `http://localhost:11434/v1` (or `MA_BASE_URL`). Bare `ma` starts an interactive REPL; `ma -p PROMPT` runs once and prints the final response. `--tools read,edit,ls,grep,bash` enables built-in tools (scoped to `--root`). `--max-turns` caps loop iterations via `MaxTurnsHook`. Supports `--think` / `--extra` for provider-specific `extra_body` fields. Does not read `.ma-config.toml` — use `resolve_profile()` directly in code for profile-based configuration.

**`__init__.py`** — Public API exports: `Agent`, `AgentConfig`, `AgentHooks`, `MaxTurnsHook`, `Message`, `ToolCall`, `FunctionCall`, `Usage`, `ToolInfo`, `tool`, `get_tool`, `list_tools`, `clear_registry`, `make_tools`, `make_web_tools`.

### Tool registration

Two ways to provide tools to `Agent`:

1. **Global registry** via `@tool` decorator — auto-discovered by all `Agent` instances when referenced by name.
2. **Per-agent tools** via `tools=[...]` constructor arg — accepts tool name strings (`"read"`, `"grep"`, `"bash"`) or `ToolInfo` objects. Name strings are resolved lazily to built-in tools scoped to `root`.

When both exist, per-agent tools take name-priority over global ones.

### Provider configuration

`.ma-config.toml` uses `[profiles.<name>]` sections (inheriting from `[profiles.default]`) and a `[config]` tree for app/tool settings. Read by `profiles.py` with priority `$MA_CONFIG_PATH` > `~/.ma-config.toml` > package-local > repo root. String values support `{{VAR_NAME}}` env var interpolation. The CLI (`minimal_cli.py`) does **not** read this file — it takes all settings as CLI flags/env vars.
