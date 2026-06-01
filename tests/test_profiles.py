"""Tests for the profile system: merging, interpolation, get_config, resolve_profile."""

from unittest.mock import patch

import pytest

from coreloop.config import AgentConfig
from coreloop.profiles import (
    _deep_merge,
    _interpolate,
    _load_merged_profile,
    get_config,
    resolve_profile,
)

# -- helpers -------------------------------------------------------------------

_SAMPLE_CONFIG = {
    "profiles": {
        "default": {
            "model": "base-model",
            "base_url": "http://localhost:11434/v1",
        },
        "fast": {
            "model": "fast-model",
            "llm_timeout": 30.0,
        },
        "with_cfg": {
            "model": "cfg-model",
            "config": {
                "tool": {"read": {"max_lines": 200}},
            },
        },
    },
    "config": {
        "tool": {"read": {"max_lines": 100, "max_bytes": 1000}},
        "ui": {"example_cli": {"max_turns": 50}},
    },
}


def _patch(cfg=_SAMPLE_CONFIG):
    """Patch _load_config to return cfg instead of reading from disk."""
    return patch("coreloop.profiles._load_config", return_value=cfg)


# -- _deep_merge ---------------------------------------------------------------


def test_deep_merge_simple_override():
    """Scalar values in the override dict replace base values."""
    assert _deep_merge({"a": 1, "b": 2}, {"b": 99, "c": 3}) == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    """Nested dicts are merged recursively, not replaced."""
    base = {"tool": {"read": {"max_lines": 100, "max_bytes": 1000}}}
    override = {"tool": {"read": {"max_lines": 200}}}
    assert _deep_merge(base, override) == {"tool": {"read": {"max_lines": 200, "max_bytes": 1000}}}


def test_deep_merge_non_dict_replaces():
    """A non-dict override value replaces a nested dict entirely."""
    assert _deep_merge({"a": {"x": 1}}, {"a": "string"}) == {"a": "string"}


def test_deep_merge_empty():
    """Merging with an empty dict returns the other dict unchanged."""
    assert _deep_merge({"a": 1}, {}) == {"a": 1}
    assert _deep_merge({}, {"a": 1}) == {"a": 1}


def test_deep_merge_does_not_mutate():
    """The base dict is not mutated by the merge."""
    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"y": 2}})
    assert base == {"a": {"x": 1}}


# -- _interpolate --------------------------------------------------------------


def test_interpolate_resolves_env_var(monkeypatch):
    """{{VAR}} placeholders are replaced with the matching env var value."""
    monkeypatch.setenv("MY_KEY", "secret")
    assert _interpolate("{{MY_KEY}}") == "secret"


def test_interpolate_partial_string(monkeypatch):
    """Placeholders embedded in a larger string are resolved in place."""
    monkeypatch.setenv("HOST", "localhost")
    assert _interpolate("http://{{HOST}}:8080") == "http://localhost:8080"


def test_interpolate_passthrough_non_string():
    """Non-string values are returned unchanged."""
    assert _interpolate(42) == 42
    assert _interpolate(None) is None
    assert _interpolate(["a"]) == ["a"]


def test_interpolate_missing_var_raises(monkeypatch):
    """A placeholder whose env var is absent raises ValueError."""
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ValueError, match="MISSING_VAR"):
        _interpolate("{{MISSING_VAR}}")


# -- _load_merged_profile ------------------------------------------------------


def test_load_merged_default():
    """The default profile is loaded with its own fields."""
    with _patch():
        result = _load_merged_profile("default")
    assert result["model"] == "base-model"


def test_load_merged_named_inherits_default():
    """A named profile inherits fields from [profiles.default]."""
    with _patch():
        result = _load_merged_profile("fast")
    assert result["model"] == "fast-model"
    assert result["base_url"] == "http://localhost:11434/v1"
    assert result["llm_timeout"] == 30.0


def test_load_merged_unknown_raises():
    """Requesting an unknown profile raises KeyError."""
    with _patch():
        with pytest.raises(KeyError, match="unknown"):
            _load_merged_profile("unknown")


def test_load_merged_global_config_present():
    """The [config] tree is included under the 'config' key of the merged result."""
    with _patch():
        result = _load_merged_profile("default")
    assert result["config"]["tool"]["read"]["max_lines"] == 100


def test_load_merged_profile_config_overrides_global_deep():
    """Per-profile [config] is deep-merged on top of the global [config]."""
    with _patch():
        result = _load_merged_profile("with_cfg")
    assert result["config"]["tool"]["read"]["max_lines"] == 200
    assert result["config"]["tool"]["read"]["max_bytes"] == 1000  # global key preserved


def test_load_merged_config_subtree_not_leaked_to_agent_fields():
    """The config subtree is present but doesn't pollute top-level agent fields."""
    with _patch():
        result = _load_merged_profile("fast")
    assert "config" in result
    assert result["config"]["ui"]["example_cli"]["max_turns"] == 50


# -- get_config ----------------------------------------------------------------


def test_get_config_from_raw():
    """A dot-path key is resolved from the merged profile's config subtree."""
    with _patch():
        raw = _load_merged_profile("default")
    assert get_config("tool.read.max_lines", raw) == 100


def test_get_config_profile_override_wins():
    """Per-profile config values take precedence over global config."""
    with _patch():
        raw = _load_merged_profile("with_cfg")
    assert get_config("tool.read.max_lines", raw) == 200


def test_get_config_missing_returns_default():
    """A missing key returns the supplied default."""
    with _patch():
        raw = _load_merged_profile("default")
    assert get_config("tool.nonexistent.key", raw, default=42) == 42


def test_get_config_global_no_profile():
    """get_config reads the global [config] tree when no raw profile is passed."""
    with _patch():
        assert get_config("ui.example_cli.max_turns") == 50


def test_get_config_path_too_deep_returns_default():
    """A dot-path that descends past a leaf returns the default."""
    with _patch():
        raw = _load_merged_profile("default")
    assert get_config("tool.read.max_lines.too.deep", raw, default=0) == 0


# -- resolve_profile -----------------------------------------------------------


def test_resolve_profile_returns_agent_config():
    """resolve_profile returns a populated AgentConfig dataclass."""
    with _patch():
        cfg = resolve_profile("default")
    assert isinstance(cfg, AgentConfig)
    assert cfg.model == "base-model"


def test_resolve_profile_ignores_cli_only_keys():
    """CLI-only keys like max_turns are not forwarded to AgentConfig."""
    cfg_with_extras = {
        "profiles": {
            "default": {
                "model": "x",
                "base_url": "http://localhost/v1",
                "max_turns": 99,
                "searxng_url": "http://s",
            },
        },
        "config": {},
    }
    with patch("coreloop.profiles._load_config", return_value=cfg_with_extras):
        cfg = resolve_profile("default")
    assert isinstance(cfg, AgentConfig)
    assert not hasattr(cfg, "max_turns")


def test_resolve_profile_interpolates_api_key(monkeypatch):
    """{{VAR}} placeholders in profile values are expanded from the environment."""
    monkeypatch.setenv("TEST_KEY", "sk-test")
    cfg_with_key = {
        "profiles": {
            "default": {
                "model": "m",
                "base_url": "http://localhost/v1",
                "api_key": "{{TEST_KEY}}",
            },
        },
        "config": {},
    }
    with patch("coreloop.profiles._load_config", return_value=cfg_with_key):
        cfg = resolve_profile("default")
    assert cfg.api_key == "sk-test"


# -- config_path / file loading ------------------------------------------------


def test_resolve_profile_custom_config_path(tmp_path):
    """A custom config_path is read instead of the default search path."""
    (tmp_path / "test.toml").write_text(
        '[profiles.default]\nmodel = "local-model"\nbase_url = "http://localhost/v1"\n'
    )
    cfg = resolve_profile("default", config_path=tmp_path / "test.toml")
    assert cfg.model == "local-model"


def test_resolve_profile_missing_file_raises(tmp_path):
    """A non-existent config_path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_profile("default", config_path=tmp_path / "nonexistent.toml")


# -- Agent.from_profile --------------------------------------------------------


def test_agent_from_profile(tmp_path):
    """Agent.from_profile builds an Agent from a named TOML profile."""
    from coreloop import Agent

    (tmp_path / "test.toml").write_text(
        '[profiles.default]\nmodel = "my-model"\nbase_url = "http://localhost/v1"\n'
    )
    agent = Agent.from_profile("default", config_path=tmp_path / "test.toml")
    assert agent.model == "my-model"


def test_agent_from_profile_default_name(tmp_path):
    """Agent.from_profile uses the 'default' profile when no name is given."""
    from coreloop import Agent

    (tmp_path / "test.toml").write_text(
        '[profiles.default]\nmodel = "default-model"\nbase_url = "http://localhost/v1"\n'
    )
    agent = Agent.from_profile(config_path=tmp_path / "test.toml")
    assert agent.model == "default-model"


def test_agent_from_profile_named(tmp_path):
    """A named profile is loaded and inherits base_url from default."""
    from coreloop import Agent

    (tmp_path / "test.toml").write_text(
        '[profiles.default]\nmodel = "base"\nbase_url = "http://localhost/v1"\n'
        '[profiles.fast]\nmodel = "fast-model"\n'
    )
    agent = Agent.from_profile("fast", config_path=tmp_path / "test.toml")
    assert agent.model == "fast-model"
    assert agent._base_url == "http://localhost/v1"  # inherited from default
