"""Provider configuration loaded from the built-in TOML file."""

from __future__ import annotations

import functools
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _find_providers_path() -> Path:
    """Locate providers.toml: env var > package-local > repo root (dev installs)."""
    if env := os.environ.get("MA_PROVIDERS_PATH"):
        return Path(env)
    pkg_local = Path(__file__).resolve().parent / "providers.toml"
    if pkg_local.exists():
        return pkg_local
    return Path(__file__).resolve().parent.parent.parent / "providers.toml"


_PROVIDERS_PATH = _find_providers_path()


@dataclass
class ProviderConfig:
    """Resolved provider: URL + optional API key."""

    base_url: str
    api_key: str | None


@functools.lru_cache(maxsize=1)
def _load_providers() -> dict[str, dict[str, Any]]:
    if not _PROVIDERS_PATH.exists():
        msg = f"Built-in providers.toml not found at {_PROVIDERS_PATH}"
        raise FileNotFoundError(msg)
    raw = _PROVIDERS_PATH.read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def resolve_provider(provider_name: str) -> ProviderConfig:
    """Look up *provider_name* in the built-in providers.toml and resolve the
    API key from the environment."""
    providers = _load_providers()
    entry = providers.get(provider_name)
    if entry is None:
        known = ", ".join(sorted(providers))
        msg = f"Unknown provider {provider_name!r}. Known: {known}"
        raise KeyError(msg)

    base_url = entry["base_url"].rstrip("/")
    env_key = entry.get("env_key_name")
    api_key: str | None = None
    if env_key:
        api_key = os.environ.get(env_key)
        if not api_key:
            msg = (
                f"Provider {provider_name!r} requires env var "
                f"{env_key!r} but it is not set"
            )
            raise ValueError(msg)
    return ProviderConfig(base_url=base_url, api_key=api_key)
