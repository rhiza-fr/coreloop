# Changelog

## [0.1.3] — 2026-06-03

### Changed

- **Web tools rewritten inline** — removed the `pvl-webtools[markdown]` dependency (which pulled in `onnxruntime` via `markitdown`). `web_search` now calls the SearXNG JSON API directly via `httpx`; `web_fetch` converts HTML to markdown using the Rust-backed `html-to-markdown` library. The `extract_mode: article` option is dropped; `markdown` (default) and `raw` remain.
- **Widened Python compatibility** — minimum version lowered from 3.14 to 3.12, adding support for 3.12 and 3.13.
- **Hardened path checks** — tightened the path-traversal validation in file tools.

### Added

- `llms.txt` — machine-readable project summary for LLM-assisted tooling.
