# Changelog

## [0.1.5] — 2026-08-28

### Fixed

- **`to_strict_schema()` deleted fields named `title` or `default`** — the rewrite
  recursed over every dict value, so those names were stripped as keywords wherever they
  appeared, including inside `properties`. The field vanished from `properties` but stayed
  in `required`, producing a schema no output could satisfy. Recursion now descends into
  the *values* of name-keyed maps (`properties`, `$defs`, `definitions`,
  `patternProperties`), so a user's field names are never treated as keywords.
- **`read` and `edit` no longer block the event loop** — their `open()` calls now run via
  `asyncio.to_thread`.

### Changed

- Cleared all `ruff check` findings across `src` and `tests`: import sorting,
  `typing.AsyncIterator` → `collections.abc`, `asyncio.TimeoutError` → `TimeoutError`, and
  `"k" in d and d["k"]` → `d.get("k")`. The catch-alls in `exec_tool`, `_safe_hook`, the
  web tools, and `_resolve_safe` are deliberate isolation boundaries and are suppressed
  with a stated reason rather than rewritten.

## [0.1.4] — 2026-08-27

### Added

- **Structured output** — `complete()`: a one-shot, tool-free call that constrains the
  model with `response_format: {"type": "json_schema"}` and returns a validated Pydantic
  instance. Accepts `str | list[Message]`, so "run the loop, then extract" composes from
  `agent.messages` with no new `Agent` API.
- `to_strict_schema()` — rewrites a Pydantic JSON Schema into the strict-legal shape
  (`additionalProperties: false`, every property required, `title`/`default` stripped).
- `stream_chat(..., schema=...)` — merges the backend's `response_format` fields into the
  request body after `build_body`, so the response cache key covers the schema.
- **`on_stream_chunk` hook** — fires for each partial `Message` as it arrives from the
  LLM stream. For progress display, per-token logging, or an early `agent.stop()`.
- **`system_nonce`** on `Agent` / `AgentConfig` — prepends a per-run nonce to the system
  prompt, defeating Ollama's cross-request prompt-cache slot reuse. Constant within a run,
  so intra-conversation caching is kept.

### Changed

- **`Backend` gains `response_format(schema)`** — breaking for external backend
  implementations. `OpenAIBackend` implements it; providers without `response_format`
  return `{}`.
- **`edit` drops `line_hint`** — an ambiguous `old_text` now asks for a longer, unique
  excerpt instead of a line number. Breaking for callers that passed `line_hint`.
- Profiles accept `tools` as a comma-separated string, not just a list.
- Tool-call logging moved from `info` to `debug`.
- `__version__` now tracks `pyproject.toml` (was stuck at `"0.1.0"`).

### Fixed

- `web_fetch` no longer raises when the HTML-to-markdown conversion yields empty content.
- **`tests/test_ollama_integration.py` was uncollectable** — the Ollama availability check
  used Python 2 `except A, B:` syntax, a `SyntaxError` that broke the whole module on
  import. The integration suite never ran on 0.1.3.
- **Multi-turn integration test replayed streaming deltas** — `test_multi_turn_conversation`
  rebuilt turn 2 from the raw `run()` output, which includes `partial=True` chunks. A
  reasoning-only delta has `content=None` and no `tool_calls`, which Ollama rejects with
  `invalid message content type: <nil>`. It now builds the next turn from `agent.messages`.
- Test-suite runtime cut — `sleep 5` shortened to `sleep 2` in the bash timeout tests, and
  the `rg` timeout test no longer sleeps 10s draining an already-killed process.

## [0.1.3] — 2026-06-03

### Changed

- **Web tools rewritten inline** — removed the `pvl-webtools[markdown]` dependency (which pulled in `onnxruntime` via `markitdown`). `web_search` now calls the SearXNG JSON API directly via `httpx`; `web_fetch` converts HTML to markdown using the Rust-backed `html-to-markdown` library. The `extract_mode: article` option is dropped; `markdown` (default) and `raw` remain.
- **Widened Python compatibility** — minimum version lowered from 3.14 to 3.12, adding support for 3.12 and 3.13.
- **Hardened path checks** — tightened the path-traversal validation in file tools.

### Added

- `llms.txt` — machine-readable project summary for LLM-assisted tooling.
