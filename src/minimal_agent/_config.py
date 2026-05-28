"""Configuration loaded from .ma-config.toml: defaults, model overrides, and providers."""

from __future__ import annotations

import functools
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_CONFIG_FILENAME = ".ma-config.toml"

# ── .env loading ──────────────────────────────────────────────────────
load_dotenv()


def config_path() -> Path:
    """Return the path of the config file currently being used."""
    return _CONFIG_PATH


def _find_config_path() -> Path:
    """Locate .ma-config.toml: MA_CONFIG_PATH env var > ~/.ma-config.toml
    > package-local > repo root (dev installs).
    """
    if env := os.environ.get("MA_CONFIG_PATH"):
        return Path(env)

    home_config = Path.home() / _CONFIG_FILENAME
    if home_config.exists():
        return home_config

    pkg_local = Path(__file__).resolve().parent / _CONFIG_FILENAME
    if pkg_local.exists():
        return pkg_local

    repo_root = Path(__file__).resolve().parent.parent.parent
    repo_new = repo_root / _CONFIG_FILENAME
    if repo_new.exists():
        return repo_new

    # Return the preferred path so error messages are meaningful
    return repo_new


_CONFIG_PATH = _find_config_path()


@dataclass
class ProviderConfig:
    """Resolved provider: URL + optional API key."""

    base_url: str
    api_key: str | None


@functools.lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        msg = (
            f"Config file not found at {_CONFIG_PATH}. "
            f"Create ~/{_CONFIG_FILENAME} or set MA_CONFIG_PATH."
        )
        raise FileNotFoundError(msg)
    raw = _CONFIG_PATH.read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


@dataclass
class DefaultConfig:
    """Resolved defaults, optionally merged with model-specific overrides."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system: str | None = None
    tools: list[str] = field(default_factory=list)
    think: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 20
    max_messages: int = 0
    searxng_url: str | None = None
    llm_timeout: float = 60.0
    tool_read_max_lines: int = 100
    tool_search_max_chars: int = 20_000
    tool_search_timeout: float = 30.0


def resolve_defaults() -> DefaultConfig:
    """Return the base ``[defaults]`` section (no model overrides applied)."""
    config = _load_config()
    return _entry_to_config(config.get("defaults", {}))


def resolve_model_config(model: str) -> DefaultConfig:
    """Return a ``DefaultConfig`` with ``[models.<model>]`` merged on top
    of ``[defaults]``.  Model-specific values completely override their
    base counterpart for that field.
    """
    config = _load_config()
    base = DefaultConfig()
    if "defaults" in config:
        base = _entry_to_config(config["defaults"])

    models_section = config.get("models", {})
    override = models_section.get(model, {})
    if override:
        base = _apply_overrides(base, override)
    return base


def _entry_to_config(entry: dict[str, Any]) -> DefaultConfig:
    return DefaultConfig(
        provider=entry.get("provider", "openai"),
        model=entry.get("model", "gpt-4o-mini"),
        system=entry.get("system"),
        tools=entry.get("tools", []),
        think=entry.get("think", False),
        extra=entry.get("extra", {}),
        max_turns=entry.get("max_turns", 20),
        max_messages=entry.get("max_messages", 0),
        searxng_url=entry.get("searxng_url"),
        llm_timeout=entry.get("llm_timeout", 60.0),
        tool_read_max_lines=entry.get("tool_read_max_lines", 100),
        tool_search_max_chars=entry.get("tool_search_max_chars", 20_000),
        tool_search_timeout=entry.get("tool_search_timeout", 30.0),
    )


def _apply_overrides(base: DefaultConfig, overrides: dict[str, Any]) -> DefaultConfig:
    """Shallow merge *overrides* on top of *base*."""
    return DefaultConfig(
        provider=overrides.get("provider", base.provider),
        model=overrides.get("model", base.model),
        system=overrides.get("system", base.system),
        tools=overrides.get("tools", base.tools),
        think=overrides.get("think", base.think),
        extra=overrides.get("extra", base.extra),
        max_turns=overrides.get("max_turns", base.max_turns),
        max_messages=overrides.get("max_messages", base.max_messages),
        searxng_url=overrides.get("searxng_url", base.searxng_url),
        llm_timeout=overrides.get("llm_timeout", base.llm_timeout),
        tool_read_max_lines=overrides.get("tool_read_max_lines", base.tool_read_max_lines),
        tool_search_max_chars=overrides.get("tool_search_max_chars", base.tool_search_max_chars),
        tool_search_timeout=overrides.get("tool_search_timeout", base.tool_search_timeout),
    )


def resolve_provider(provider_name: str) -> ProviderConfig:
    """Look up *provider_name* under ``[providers]`` in the config and
    resolve the API key from the environment."""
    config = _load_config()
    providers = config.get("providers", {})
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
