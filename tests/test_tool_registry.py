"""Tests for ToolInfo.__call__, the @tool decorator, and the global registry."""

import pytest

from coreloop import tool
from coreloop.tool_registry import ToolInfo, clear_registry, get_tool, list_tools


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate the global registry for each test."""
    yield
    clear_registry()


# -- ToolInfo.__call__ ---------------------------------------------------------


@pytest.mark.asyncio
async def test_toolinfo_call_delegates_to_fn():
    """Calling a ToolInfo directly invokes the wrapped function."""
    async def fn(x: int) -> str:
        return f"got {x}"

    ti = ToolInfo(
        name="t", description="", parameters={"type": "object", "properties": {}}, fn=fn
    )
    result = await ti(x=42)
    assert result == "got 42"


# -- @tool decorator -----------------------------------------------------------


def test_tool_decorator_registers_function():
    """@tool adds the function to the global registry."""
    @tool
    async def my_fn() -> str:
        """My function."""
        return "hi"

    info = get_tool("my_fn")
    assert info is not None
    assert info.name == "my_fn"


def test_tool_decorator_with_custom_name():
    """@tool(name=...) registers under the custom name."""
    @tool(name="renamed")
    async def original_name() -> str:
        """A function."""
        return "ok"

    assert get_tool("renamed") is not None
    assert get_tool("original_name") is None


def test_tool_decorator_duplicate_raises():
    """Registering the same name twice raises ValueError without allow_override."""
    @tool
    async def dup() -> str:
        """First."""
        return "a"

    with pytest.raises(ValueError, match="already registered"):
        @tool
        async def dup() -> str:  # noqa: F811
            """Second."""
            return "b"


def test_tool_decorator_allow_override():
    """allow_override=True silently replaces the existing registration."""
    @tool
    async def overridable() -> str:
        """First."""
        return "first"

    @tool(allow_override=True)
    async def overridable() -> str:  # noqa: F811
        """Second."""
        return "second"

    assert get_tool("overridable") is not None


# -- list_tools / clear_registry -----------------------------------------------


def test_list_tools_returns_registered():
    """list_tools returns all currently registered tools."""
    @tool
    async def aaa() -> str:
        """AAA."""
        return "a"

    @tool
    async def bbb() -> str:
        """BBB."""
        return "b"

    names = {t.name for t in list_tools()}
    assert "aaa" in names
    assert "bbb" in names


def test_clear_registry_removes_all():
    """clear_registry empties the global registry."""
    @tool
    async def temp() -> str:
        """Temp."""
        return "t"

    clear_registry()
    assert list_tools() == []
