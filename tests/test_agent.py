"""Tests for Agent construction and _resolve_tools name resolution."""

import tempfile

import pytest

from coreloop import Agent
from coreloop.agent import _resolve_tools
from coreloop.tool_registry import ToolInfo, clear_registry, tool


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the global tool registry for each test."""
    yield
    clear_registry()


def _noop_info(name: str) -> ToolInfo:
    """Build a minimal ToolInfo with the given name."""
    async def fn() -> str:
        return "ok"
    return ToolInfo(name=name, description="", parameters={"type": "object", "properties": {}}, fn=fn)


# -- _resolve_tools ------------------------------------------------------------


def test_resolve_toolinfo_passthrough():
    """A ToolInfo passed directly is kept as-is."""
    ti = _noop_info("my_tool")
    result = _resolve_tools([ti], root=None)
    assert result["my_tool"] is ti


def test_resolve_file_tool_names():
    """String names for built-in file tools are resolved to scoped ToolInfo objects."""
    with tempfile.TemporaryDirectory() as root:
        result = _resolve_tools(["read", "ls", "edit", "grep"], root=root)
    assert set(result.keys()) == {"read", "ls", "edit", "grep"}


def test_resolve_bash_tool_name():
    """The 'bash' string name resolves to a bash ToolInfo."""
    with tempfile.TemporaryDirectory() as root:
        result = _resolve_tools(["bash"], root=root)
    assert "bash" in result


def test_resolve_global_registered_tool():
    """A globally @tool-registered function is resolved by name."""
    @tool
    async def my_custom_tool() -> str:
        """Custom tool."""
        return "custom"

    result = _resolve_tools(["my_custom_tool"], root=None)
    assert result["my_custom_tool"].name == "my_custom_tool"


def test_resolve_unknown_name_raises():
    """An unknown tool name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown tool"):
        _resolve_tools(["no_such_tool_xyz"], root=None)


def test_resolve_later_entry_wins_on_collision():
    """When the same name appears twice, the last ToolInfo wins."""
    ti1 = _noop_info("t")
    ti2 = _noop_info("t")
    result = _resolve_tools([ti1, ti2], root=None)
    assert result["t"] is ti2


# -- Agent construction --------------------------------------------------------


def test_agent_default_stopped_false():
    """A freshly constructed Agent is not stopped."""
    agent = Agent(model="test")
    assert not agent.stopped


def test_agent_stop_sets_stopped():
    """Calling stop() marks the agent as stopped."""
    agent = Agent(model="test")
    agent.stop()
    assert agent.stopped


def test_agent_model_is_stored():
    """The model name passed to Agent is accessible via agent.model."""
    agent = Agent(model="my-model")
    assert agent.model == "my-model"
