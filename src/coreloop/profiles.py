"""Configuration loaded from coreloop.toml: named profiles with default merging.

Profile resolution:
  1. Load [profiles.default] as the base.
  2. Merge [profiles.<name>] on top (named keys win).
  3. Deep-merge [config] global settings with any [profiles.<name>.config] overrides.
  4. Resolve {{VAR_NAME}} in any string value from the environment.
  5. Strip non-AgentConfig keys and return AgentConfig.

The special name "default" is the base -- every other profile inherits from it.
[config] is the global settings tree; profiles can override any sub-key.
"""

import functools
import os
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import AgentConfig

from dotenv import load_dotenv

_CONFIG_FILENAME = ".coreloop.toml"
_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")

load_dotenv()


def config_path() -> Path:
    """Return the path of the config file currently being used."""
    return _CONFIG_PATH


def _find_config_path() -> Path:
    if env := os.environ.get("CORELOOP_CONFIG"):
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
    return repo_new


_CONFIG_PATH = _find_config_path()


@functools.lru_cache(maxsize=8)
def _load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _CONFIG_PATH
    if not p.exists():
        msg = (
            f"Config file not found at {p}. "
            f"Create ~/{_CONFIG_FILENAME} or set CORELOOP_CONFIG."
        )
        raise FileNotFoundError(msg)
    return tomllib.loads(p.read_bytes().decode("utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return base with override applied recursively. Dicts merge; all other types replace."""
    result = dict(base)
    for k, v in override.items():
        result[k] = (
            _deep_merge(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else v
        )
    return result


def _interpolate(value: Any) -> Any:
    """Resolve {{VAR_NAME}} in string values from the environment."""
    if not isinstance(value, str):
        return value

    def _sub(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            msg = f"Environment variable {var!r} referenced in config is not set"
            raise ValueError(msg)
        return val

    return _TEMPLATE_RE.sub(_sub, value)


def _interpolate_strings(data: dict[str, Any]) -> dict[str, Any]:
    return {k: _interpolate(v) for k, v in data.items()}


def _load_merged_profile(name: str, config_path: Path | str | None = None) -> dict[str, Any]:
    """Return the raw merged profile dict (default base + named overlay + config tree).

    Top-level profile keys are shallow-merged (named profile wins over default).
    The [config] tree is deep-merged: global < default profile < named profile.
    Includes all keys -- callers that only need AgentConfig should use resolve_profile().
    """
    raw = _load_config(Path(config_path) if config_path else None)
    profiles = raw.get("profiles", {})
    global_cfg = raw.get("config", {})

    base = dict(profiles.get("default", {}))
    base_cfg = base.pop("config", {})

    if name != "default":
        if name not in profiles:
            known = ", ".join(sorted(k for k in profiles if k != "default"))
            msg = f"Unknown profile {name!r}. Known: {known or '(none)'}"
            raise KeyError(msg)
        overlay = dict(profiles[name])
        overlay_cfg = overlay.pop("config", {})
        merged = {**base, **overlay}
        profile_cfg = _deep_merge(base_cfg, overlay_cfg)
    else:
        merged = base
        profile_cfg = base_cfg

    merged["config"] = _deep_merge(global_cfg, profile_cfg)
    return merged


def get_config(
    key: str,
    raw_profile: dict[str, Any] | None = None,
    default: Any = None,
    *,
    config_path: Path | str | None = None,
) -> Any:
    """Look up a dot-path key in the [config] tree.

    get_config("tool.web_search.url", raw)   -- uses merged profile config
    get_config("tool.web_search.url")         -- uses global [config] only

    {{VAR_NAME}} interpolation is applied to string values.
    Returns default if the key is absent or the path is invalid.
    """
    if raw_profile is not None:
        node: Any = raw_profile.get("config", {})
    else:
        node = _load_config(Path(config_path) if config_path else None).get("config", {})

    for part in key.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(part)
        if node is None:
            return default

    return _interpolate(node) if isinstance(node, str) else node


def resolve_profile(
    name: str = "default", *, config_path: Path | str | None = None
) -> "AgentConfig":
    """Return an AgentConfig for the named profile.

    Merges [profiles.default] as base, then [profiles.<name>] on top.
    {{VAR_NAME}} in any string value is resolved from the environment.
    Unknown keys (config, max_turns, searxng_url, ...) are silently ignored.
    """
    import dataclasses

    from .config import AgentConfig

    raw = _interpolate_strings(_load_merged_profile(name, config_path=config_path))
    known = {f.name for f in dataclasses.fields(AgentConfig)}
    return AgentConfig(**{k: v for k, v in raw.items() if k in known})
