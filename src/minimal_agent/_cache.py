"""Disk-based cache for LLM responses, keyed by a hash of the request."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache


def make_cache(path: str | Path) -> diskcache.Cache:
    """Open (or create) a disk cache at the given directory path."""
    return diskcache.Cache(str(path))


def request_key(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    extra_body: dict[str, Any] | None,
) -> str:
    """Return a hex SHA-256 key for the given request parameters."""
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    if extra_body:
        payload["extra_body"] = extra_body
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
