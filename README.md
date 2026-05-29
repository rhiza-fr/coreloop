# minimal-agent

Provides a lightweight async tool-calling agent core, fully interruptable and observable.
Usable as library - core AsyncIterator[Message], hooks, logging.
Any OpenAI-compatible API via httpx
Minimal REPL
Configurable tools: read, edit, ls, search
web_search and web_fetch via the [web] extra.

No MCP, no SKILLS


## Install

```bash
pip install minimal-agent
# with web tools (web_search, web_fetch):
pip install "minimal-agent[web]"
```

## CLI

```bash
# Interactive REPL
ma --model gpt-4o-mini --provider openai

# One-shot
ma -p "Summarise this repo" --model gpt-4o-mini

# With file tools
ma --tools read,ls,search,edit --root .

# With web tools (requires [web] and a SearXNG url)
ma --tools web_search,web_fetch --searxng-url http://localhost:8080

# With thinking enabled
ma -p "Solve this step by step" --think
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --prompt` | — | Run once and print result (non-interactive) |
| `-m, --model` | `gpt-4o-mini` | Model name |
| `--provider` | `openai` | Provider (see below) |
| `-s, --system` | — | System prompt |
| `--tools` | — | Comma-separated tools: `read,ls,edit,search,web_search,web_fetch` |
| `-r, --root` | cwd | Allowed root directory for file tools |
| `--searxng-url` | `$SEARXNG_URL` | SearXNG base URL for web tools |
| `-t, --timeout` | `60.0` | Timeout in seconds for LLM and tool calls |
| `-n, --max-turns` | `20` | Maximum agent loop iterations |
| `--think/--no-think` | off | Enable (`medium`) or disable (`none`) reasoning_effort |
| `-e, --extra` | — | Extra JSON body params merged into the API request |
| `--json` | off | Output all messages as JSONL |
| `-V, --version` | — | Show version and exit |

### REPL commands

Once the interactive REPL starts:

| Command | Description |
|---------|-------------|
| `/quit` or `/exit` or `/q` | Exit the REPL |
| `/stop` | Cancel the current agent run mid-turn |
| `/new` | Clear conversation history and start fresh |
| `/root <path>` | Change the allowed root for file tools (if `--tools` was set) |

## Providers

Providers are configured in `.ma-config.toml`. Resolution order (first found wins):

1. `$MA_CONFIG_PATH` env var
2. `~/.ma-config.toml`
3. Package-local `src/minimal_agent/.ma-config.toml`
4. Repo root `.ma-config.toml` (for dev installs)

On first run, `ma` auto-creates `~/.ma-config.toml` from the shipped default.

Two top-level sections:

- `[defaults]` — default provider, model, tools, and other settings
- `[providers.<name>]` — each provider with `base_url` and optional `env_key_name`
- `[models.<name>]` — optional per-model overrides merged on top of `[defaults]`

Example:

```toml
[defaults]
provider = "openai"
model = "gpt-4o-mini"
tools = ["read", "edit", "ls", "search"]
# think = false
# extra = {"reasoning_effort": "medium"}
# max_turns = 20
# searxng_url = "http://localhost:8888"

[providers.openai]
base_url = "https://api.openai.com/v1"
env_key_name = "OPENAI_API_KEY"

[providers.ollama]
base_url = "http://localhost:11434/v1"

# Per-model overrides — any field from [defaults] can be overridden
[models."gpt-4o"]
think = true
max_turns = 30
```

| Provider | Env var |
|----------|---------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | (none) |

## Tools

### File tools (`make_tools`)

Enabled with `--tools read,ls,edit,search`. All tools are scoped to `--root` and reject path traversal.

| Tool | Description |
|------|-------------|
| `read` | Read a text file with optional `offset`/`limit` (line numbers) |
| `ls` | List a directory |
| `edit` | Replace an exact string in a file (single occurrence, with optional `line_hint`) |
| `search` | Regex search via `rg`. Supports `type`, `after_context`, `files_with_matches`. Output capped at 20 000 chars. |

### Web tools (`make_web_tools`)

Requires `pip install "minimal-agent[web]"` and a running [SearXNG](https://docs.searxng.org/) instance.

```bash
# Docker quick-start
docker run -d -p 8080:8080 searxng/searxng

ma --tools web_search,web_fetch --searxng-url http://localhost:8080 -p "Latest news on X"
# or set the env var instead:
export SEARXNG_URL=http://localhost:8080
ma --tools web_search,web_fetch -p "Latest news on X"
```

| Tool | Description |
|------|-------------|
| `web_search` | Search via SearXNG. Returns titles, URLs, snippets. Supports `max_results`, `domain_filter`, `recency` (`all_time`, `day`, `week`, `month`, `year`). |
| `web_fetch` | Fetch a URL. Returns content in `extract_mode`: `markdown` (default), `article`, `raw`, or `metadata`. |

## Library

```python
from minimal_agent import Agent, Message, tool, make_tools, make_web_tools

# Register a custom tool globally
@tool
async def shout(text: str) -> str:
    """Return text in uppercase."""
    return text.upper()

agent = Agent(
    model="gpt-4o-mini",
    provider="openai",
    system="You are helpful.",
    tools=make_tools(allowed_root="/tmp/sandbox"),
)

async for msg in agent.run([Message(role="user", content="List the files here, then read README.md")]):
    if msg.content:
        print(msg.content)

# Full message history available after run
print(agent.messages)
```

### `Agent`

```python
Agent(
    model: str,
    provider: str = "openai",
    system: str | None = None,
    tools: list[ToolInfo] | None = None,   # per-agent (takes priority over global)
    timeout: float = 60.0,
    hooks: AgentHooks | None = None,
    extra_body: dict | None = None,
    cache_dir: Path | str | None = "~/.cache/minimal-agent",  # None disables caching
)
```

The agent core has no built-in turn limit. To bound a run, attach a hook that calls `agent.stop()` — see `MaxTurnsHook` in `examples/hooks.py` (this is how the CLI's `--max-turns` is implemented).

`agent.run(messages)` is an async generator that yields `Message` objects as they stream. It accepts an optional `usage` keyword argument — a mutable `Usage` object that cumulative token counts are added to after each LLM turn (requires provider support).

| Method | Description |
|--------|-------------|
| `stop()` | Finish the current turn cleanly, then exit the loop. Safe to call from a tool or hook; `on_after_agent` still fires. |
| `abort()` | Halt immediately, abandoning in-flight tools. `on_after_agent` is **not** called. |
| `reset()` | Clear message history and reset the stop flag |
| `stopped` (property) | Whether `stop()` or `abort()` has been called |
| `messages` (property) | Shallow copy of the full chat history from the last `run()` |

**Restart pattern** — pass `agent.messages` to a second `run()` to keep history:

```python
async for msg in agent.run([Message(role="user", content="Hi")]):
    ...

# Restart with a different model, keeping the conversation
agent.model = "better-model"
async for msg in agent.run(agent.messages):
    ...
```

### `Message`

Pydantic model matching the OpenAI chat format:

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | `"user"`, `"assistant"`, `"system"`, or `"tool"` |
| `content` | `str \| None` | Text content (`None` when `tool_calls` is set) |
| `tool_calls` | `list[ToolCall] \| None` | Tool calls emitted by the assistant |
| `tool_call_id` | `str \| None` | Tool call ID (for `role="tool"` messages) |
| `name` | `str \| None` | Tool name (for `role="tool"` messages) |
| `reasoning` | `str \| None` | Streaming-only field from thinking models (Qwen3, DeepSeek). Omitted from conversation history sent back to the API. |
| `partial` | `bool` | `True` for streaming delta chunks; `False` for the final assembled message |
| `usage` | `Usage \| None` | Token usage reported by the model |
| `duration` | `float \| None` | Tool execution duration in seconds |
| `model` | `str \| None` | Model name that generated this message |
| `timestamp` | `datetime` | When the message was created (UTC) |

### `ToolInfo`

```python
@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Coroutine[Any, Any, str]]
```

### `ToolCall` / `FunctionCall`

```python
class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class FunctionCall(BaseModel):
    name: str = ""
    arguments: str = ""  # JSON-encoded
```

### `Usage`

```python
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

### Custom tools

Use the `@tool` decorator for global tools, or pass closures directly as `ToolInfo` for per-agent state (see `make_tools` source for the pattern).

```python
from minimal_agent import tool

@tool
async def read_env(name: str) -> str:
    """Read an environment variable."""
    import os
    return os.environ.get(name, "(not set)")
```

Tools are auto-discovered by all `Agent` instances once registered. To register under a different name or with a description:

```python
@tool(name="my_read", description="Read a file")
async def read(path: str) -> str:
    ...

@tool(allow_override=True)  # re-register even if a tool with that name exists
async def override(name: str) -> str:
    ...
```

### Registry helpers

```python
from minimal_agent import clear_registry
clear_registry()  # remove all registered tools (useful in tests)
```

### Caching

LLM responses are disk-cached by default (in `~/.cache/minimal-agent/`) to avoid re-sending identical requests. Use `cache_dir` to choose a location, or `cache_dir=None` to disable:

```python
agent = Agent(model="...", cache_dir="/tmp/my-cache")  # custom location
agent = Agent(model="...", cache_dir=None)             # caching disabled
```
