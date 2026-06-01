"""Disk-based cache for LLM responses, keyed by a hash of the request."""

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache


def make_cache(path: str | Path) -> diskcache.Cache:
    """Open (or create) a disk cache at the given directory path."""
    return diskcache.Cache(str(path))


def request_key(base_url: str, body: dict[str, Any]) -> str:
    """Return a hex SHA-256 key for the full request (endpoint + body)."""
    payload = {"base_url": base_url, "body": body}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
