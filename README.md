# minimal-agent

A lightweight async tool-calling agent for any OpenAI-compatible API (via `httpx`).
The core is an async generator loop that streams `Message` objects; you observe and
intercept it via lifecycle hooks. Usable as a library or through a minimal CLI.

The minimal core imposes no forced overhead, so it pairs well with small local
models; early exit via hooks and `stop()` makes high-throughput batch work fast and
keeps token costs down.

Built-in tools: path-scoped `read`, `ls`, `edit`, `grep`; a `bash` tool with
best-effort guardrails (not a security sandbox — see below); optional `web_search`
and `web_fetch` (via the `[web]` extra).

- **Observability** — [hook into every stage of the loop](#hooks): before/after each turn, LLM call, and tool execution
  — examples: [streaming](examples/example6_streaming.py) · [raw message stream](examples/example7_raw_messages.py) · [logging & timing hooks](examples/example8_hooks.py)
- **Loop control** — [stop cleanly, abort immediately, or inject responses mid-run](#agent): `stop()`, `abort()`, `on_before_llm`
  — examples: [lifecycle control](examples/example3_agent_lifecycle.py) · [intercept & replace results](examples/example9_intrusive_hooks.py)
- **Extensibility** — [register custom async tools with `@tool`](#custom-tools): inferred JSON Schema, callable directly or by name
  — examples: [custom tools](examples/example5_customtools.py) · [subagents](examples/example12_subagents.py)

## Install

```bash
pip install minimal-agent
pip install "minimal-agent[web]"   # adds web_search and web_fetch
```

## Library quick-start

```python
import asyncio
from minimal_agent import Agent, Message

agent = Agent(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    tools=["read", "ls", "grep"],
    root="/tmp/sandbox",
)

async def main():
    async for msg in agent.run([Message(role="user", content="What files are here?")]):
        if msg.role == "assistant" and not msg.partial and msg.content:
            print(msg.content)

asyncio.run(main())
```

Or load settings from a named profile in `~/minimal-agent.toml`:

```python
agent = Agent.from_profile("openai")
```

## CLI

`ma` is a REPL / one-shot runner with profile support. On first run it copies
the bundled `minimal-agent.toml` to `~/minimal-agent.toml` — edit that file to set
your default model, tools, and provider credentials.

```bash
# Interactive REPL using the default profile (Ollama)
ma

# One-shot with a named profile
ma --profile openai -p "Summarise this repo"

# Override model and enable file tools
ma --profile openai --model gpt-4o --tools read,ls,grep --root .

# Bypass profiles entirely
ma --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o-mini -p "Hello"

# Enable reasoning
ma --think -p "Explain this step by step" --model qwen3-14b
```

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `default` | Named profile from `~/minimal-agent.toml` |
| `-m, --model` | profile value | Model name — overrides profile (`MINIMAL_AGENT_MODEL`) |
| `--base-url` | profile value | API base URL — overrides profile (`MINIMAL_AGENT_BASE_URL`) |
| `--api-key` | profile value | API key — overrides profile (`MINIMAL_AGENT_API_KEY`) |
| `-s, --system` | — | System prompt |
| `--tools` | profile value | Comma-separated: `read,ls,edit,grep,bash,web_search,web_fetch` |
| `-r, --root` | cwd | Allowed root directory for file tools |
| `--searxng-url` | `$SEARXNG_URL` | SearXNG base URL for web tools |
| `-t, --llm-timeout` | profile value | Asyncio wall-clock timeout per LLM turn (seconds) |
| `--tool-timeout` | profile value | Hard timeout per tool call (seconds) |
| `--http-request-timeout` | profile value | httpx per-chunk read timeout (seconds) |
| `--cache-dir` | profile value | LLM response cache directory |
| `--no-cache` | off | Disable response caching |
| `-e, --extra` | — | Extra JSON merged into the API request body |
| `--think/--no-think` | off | Set `reasoning_effort` to `medium` / `none` |
| `-n, --max-turns` | `20` | Maximum loop iterations |
| `-p, --prompt` | — | Run once and print final response |
| `--json` | off | Output all non-partial messages as JSONL |
| `-l, --log-level` | — | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

REPL commands: `/quit` `/exit` `/q` to exit; `/new` to clear history;
`/model <name>` to switch models; `/root <path>` to change the file-tool root.

## `Agent`

```python
Agent(
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str | None = None,
    system: str | None = None,
    tools: list[str | ToolInfo] | None = None,
    root: str | Path | None = None,
    http_request_timeout: float = 300.0,  # httpx per-chunk read timeout
    tool_timeout: float = 360.0,          # hard wall per tool call
    llm_timeout: float = 300.0,           # asyncio wall for the entire LLM turn
    hooks: AgentHooks | None = None,
    llm_extra_body: dict | None = None,
    cache_dir: Path | str | None = "~/.cache/minimal-agent-llm-cache",
)
```

`tools` accepts built-in names (`"read"`, `"ls"`, `"edit"`, `"grep"`, `"bash"`,
`"web_search"`, `"web_fetch"`), names of `@tool`-registered functions, or `ToolInfo`
objects. File tools are scoped to `root`; an unknown name raises `ValueError`. An agent
has exactly the tools you list — there is no implicit inclusion of the global registry.

`run(messages)` is an async generator. Partial streaming chunks have `partial=True`;
the final assembled message for each LLM turn has `partial=False`. Pass
`usage=Usage()` to accumulate token counts across turns.

| Method / property | Description |
|---|---|
| `run(messages, *, usage=None)` | Run the agent loop, yielding `Message` objects |
| `stop()` | Finish the current turn cleanly, then exit. Safe from a hook or tool. |
| `abort()` | Cancel immediately; `on_after_agent` is not called |
| `reset()` | Clear history and stop flag |
| `stopped` | `True` after `stop()` or `abort()` |
| `messages` | Shallow copy of full chat history from the last `run()` |

**Restart pattern** — pass `agent.messages` to keep history across runs:

```python
async for msg in agent.run([Message(role="user", content="Hello")]):
    ...

agent.model = "stronger-model"
async for msg in agent.run(agent.messages + [Message(role="user", content="Now do X")]):
    ...
```

## Built-in tools

All file tools reject path traversal and are scoped to `root`.

| Tool | Description |
|------|-------------|
| `read` | Read a text file; `offset`/`limit` for paging (1-based line numbers) |
| `ls` | List a directory |
| `edit` | Replace an exact string in a file; fails on ambiguous matches |
| `grep` | Regex search via `rg`; supports `type`, `after_context`, `files_with_matches` |
| `bash` | Run arbitrary shell commands via `bash -c`; kills the entire process group on timeout |

> **`bash` is not sandboxed.** It runs whatever the model sends with your full
> user privileges. The `workdir` is scoped to `root`, but a command can still read,
> write, or delete anything your account can reach, and make network calls. The
> dangerous-pattern blocklist (e.g. `rm -rf /`, `mkfs`, fork bombs) is a speed bump
> against obvious accidents, **not** a security boundary — trivial variants slip
> through (`curl … | bash`, `python -c "…"`, unusual flag orders). Only enable
> `bash` for models and prompts you trust, and prefer running in a container or VM.

Web tools require `pip install "minimal-agent[web]"` and a running
[SearXNG](https://docs.searxng.org/) instance:

```bash
docker run -d -p 8080:8080 searxng/searxng
```

| Tool | Description |
|------|-------------|
| `web_search` | Search via SearXNG; returns titles, URLs, snippets; supports `max_results`, `domain_filter`, `recency` |
| `web_fetch` | Fetch a URL; `extract_mode`: `markdown` (default), `article`, `raw`, `metadata` |

## Custom tools

```python
from minimal_agent import tool

@tool
async def read_env(name: str) -> str:
    """Read an environment variable."""
    import os
    return os.environ.get(name, "(not set)")

# Attach by name or pass the ToolInfo object directly
agent = Agent(model="...", tools=["read_env"])
agent = Agent(model="...", tools=[read_env])  # equivalent
```

`@tool` infers JSON Schema from type annotations (`str`, `int`, `float`, `bool`,
`list[T]`, `dict`, `Optional[T]`). Unrecognised types fall back to `string` with a
warning. Override name or description:

```python
@tool(name="my_read", description="Read a local file")
async def _impl(path: str) -> str:
    ...

@tool(allow_override=True)   # re-register if a tool with that name already exists
async def read_env(name: str) -> str:
    ...
```

The decorated object is both a `ToolInfo` and directly callable (`await my_tool(...)`
works alongside registration).

Registry helpers:

```python
from minimal_agent import list_tools, get_tool, clear_registry

list_tools()          # list[ToolInfo] — all globally registered tools
get_tool("read_env")  # ToolInfo | None
clear_registry()      # remove all tools (useful in tests)
```

## Hooks

Subclass `AgentHooks` to observe or intercept any stage of the loop:

```python
from minimal_agent import AgentHooks

class LogHook(AgentHooks):
    async def on_before_tool(self, agent, name, args):
        print(f"→ {name}({args})")
        return None  # None = run the tool; return str to inject a result instead

agent = Agent(..., hooks=LogHook())
```

Hook firing order per turn:

```
on_before_turn
on_before_llm   → return Message to skip the LLM call entirely
  <LLM streams>
on_after_llm    → return Message to replace before appending to history
  for each tool (in parallel):
    on_before_tool  → return str to skip tool execution
      <tool runs>
    on_after_tool   → return str to replace result in history
on_after_turn
```

`on_before_agent` / `on_after_agent` bracket the entire `run()` call.
`on_after_agent` is **not** called after `abort()`. Hook exceptions are logged and
swallowed — they cannot crash the agent.

`MaxTurnsHook(n)` calls `agent.stop()` after `n` turns; the counter resets on each
`run()`, so it enforces a per-run limit, not a lifetime one.

## Config profiles

`Agent.from_profile("openai")` builds an `AgentConfig` from `~/minimal-agent.toml`
(or `$MINIMAL_AGENT_CONFIG`). Every profile inherits from `[profiles.default]`; named
profiles override individual keys.

```toml
[profiles.default]
base_url = "http://localhost:11434/v1"
model    = "qwen3.5:9b"

[profiles.openai]
base_url = "https://api.openai.com/v1"
api_key  = "{{OPENAI_API_KEY}}"
model    = "gpt-4o-mini"
tools    = ["read", "ls", "grep"]

[profiles.together]
base_url = "https://api.together.xyz/v1"
api_key  = "{{TOGETHER_API_KEY}}"
```

`{{VAR_NAME}}` in any string value is interpolated from the environment. Unknown keys
are silently ignored. The shipped `src/minimal_agent/minimal-agent.toml` includes
pre-configured profiles for Ollama, OpenAI, Groq, DeepSeek, Together, and OpenRouter.

`AgentConfig` is a dataclass mirroring the `Agent` constructor (excluding hooks). Use
`dataclasses.replace(cfg, model="other")` to derive variants, or `Agent.from_config(cfg)`
to construct an agent from one.

## `Message`

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | `"user"`, `"assistant"`, `"system"`, `"tool"` |
| `content` | `str \| None` | Text content; `None` when `tool_calls` is set |
| `tool_calls` | `list[ToolCall] \| None` | Tool calls emitted by the assistant |
| `tool_call_id` | `str \| None` | Links a `"tool"` message to its call |
| `name` | `str \| None` | Tool name on `role="tool"` messages |
| `reasoning` | `str \| None` | Thinking-model scratchpad (Qwen3, DeepSeek). Not sent back to the API. |
| `partial` | `bool` | `True` for streaming delta chunks |
| `usage` | `Usage \| None` | Token counts from the model |
| `model` | `str \| None` | Model name reported by the API |
| `duration` | `float \| None` | Seconds elapsed for the LLM turn or tool call |
| `timestamp` | `datetime` | UTC creation time |

## Caching

LLM responses are disk-cached by default (`~/.cache/minimal-agent-llm-cache/`), keyed
by SHA-256 of (model, messages, tools, extra_body). This replays identical requests
without hitting the API — useful during development. Pass `cache_dir=None` to disable.

```python
agent = Agent(model="...", cache_dir="/tmp/my-cache")  # custom location
agent = Agent(model="...", cache_dir=None)             # disabled
```
