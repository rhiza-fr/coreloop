# minimal-agent

A minimal, dependency-light LLM agent loop built on any OpenAI-compatible API.

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

# With web tools (requires SearXNG)
ma --tools web_search,web_fetch --searxng-url http://localhost:8080

# With reasoning enabled
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
| `-M, --max-messages` | `0` (unlimited) | Stop after N yielded messages |
| `--think` | off | Enable `reasoning_effort=medium` |
| `-e, --extra` | — | Extra JSON body params merged into the API request |
| `--json` | off | Output all messages as JSONL |

## Providers

Providers are configured in `providers.toml` at the repo root. Each entry specifies a `base_url` and an optional `env_key_name` for the API key.

| Provider | Env var |
|----------|---------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | (none) |

Point `MA_PROVIDERS_PATH` at a custom TOML file to override or extend the built-in list.

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
from minimal_agent import Agent, tool, make_tools, make_web_tools

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

async for msg in agent.run("List the files here, then read README.md"):
    if msg.content:
        print(msg.content)

# Full conversation history available after run
print(agent.conversation)
```

### `Agent`

```python
Agent(
    model: str,
    provider: str = "openai",
    system: str | None = None,
    tools: list[ToolInfo] | None = None,   # per-agent (takes priority)
    timeout: float = 60.0,
    max_turns: int = 20,
    max_messages: int = 0,
    extra_body: dict | None = None,
)
```

`agent.run(prompt)` is an async generator that yields `Message` objects as they stream. Call `agent.stop()` to cancel mid-run. After completion, `agent.conversation` holds the full history for reuse or handoff.

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

Tools are auto-discovered by all `Agent` instances once registered.
