"""Tests for exec_tool and run_tool: argument validation, hooks, timeout, and error paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from coreloop._tool_execution import exec_tool, run_tool
from coreloop.hooks import AgentHooks
from coreloop.tool_registry import ToolInfo
from coreloop.types import FunctionCall, ToolCall


def _make_tool(name: str, fn=None, *, required: list[str] | None = None) -> ToolInfo:
    """Build a ToolInfo for testing."""
    if fn is None:
        async def fn(**kwargs) -> str:
            return "ok"
    props = {p: {"type": "string"} for p in (required or [])}
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return ToolInfo(name=name, description="", parameters=schema, fn=fn)


def _make_tc(name: str, args: str = "{}") -> ToolCall:
    """Build a ToolCall with the given name and JSON args string."""
    return ToolCall(id="c1", function=FunctionCall(name=name, arguments=args))


def _make_agent(tool: ToolInfo | None = None, hooks: AgentHooks | None = None) -> MagicMock:
    """Build a minimal Agent mock that resolves a single optional tool."""
    agent = MagicMock()
    agent.hooks = hooks or AgentHooks()
    agent.tool_timeout = 5.0
    agent._resolve_tool = MagicMock(return_value=tool)
    agent._all_tools = MagicMock(return_value=[tool] if tool else [])
    return agent


# -- run_tool ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_executes_and_returns_result():
    """A valid call returns the tool's return value as a string."""
    tool = _make_tool("t", fn=AsyncMock(return_value="hello"))
    result = await run_tool(tool, {}, timeout=5.0)
    assert result == "hello"


@pytest.mark.asyncio
async def test_run_tool_missing_required_arg():
    """Missing a required argument returns an error without calling the tool."""
    called = []

    async def fn(path: str) -> str:
        called.append(path)
        return "ok"

    tool = _make_tool("t", fn=fn, required=["path"])
    result = await run_tool(tool, {}, timeout=5.0)
    assert "missing required" in result
    assert not called


@pytest.mark.asyncio
async def test_run_tool_unknown_arg():
    """Passing an unknown argument returns an error without calling the tool."""
    called = []

    async def fn() -> str:
        called.append(True)
        return "ok"

    tool = _make_tool("t", fn=fn)
    result = await run_tool(tool, {"extra": "x"}, timeout=5.0)
    assert "unexpected arguments" in result
    assert not called


@pytest.mark.asyncio
async def test_run_tool_timeout():
    """A tool that exceeds its timeout returns a timeout error."""
    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    tool = _make_tool("t", fn=slow)
    result = await run_tool(tool, {}, timeout=0.05)
    assert "timed out" in result


@pytest.mark.asyncio
async def test_run_tool_exception_is_caught():
    """An exception raised by the tool is caught and returned as an error string."""
    async def boom() -> str:
        raise ValueError("kaboom")

    tool = _make_tool("t", fn=boom)
    result = await run_tool(tool, {}, timeout=5.0)
    assert "Error in tool" in result
    assert "kaboom" in result


# -- exec_tool -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_tool_bad_json_args():
    """Malformed JSON arguments return an error without calling the tool."""
    tc = ToolCall(id="c1", function=FunctionCall(name="t", arguments="{bad json}"))
    agent = _make_agent()
    _, result, _ = await exec_tool(tc, agent)
    assert "failed to parse arguments" in result


@pytest.mark.asyncio
async def test_exec_tool_unknown_tool():
    """Requesting an unknown tool returns an error naming the tool."""
    tc = _make_tc("no_such_tool")
    agent = _make_agent(tool=None)
    _, result, _ = await exec_tool(tc, agent)
    assert "unknown tool" in result


@pytest.mark.asyncio
async def test_exec_tool_calls_tool_and_returns_result():
    """exec_tool runs the tool and returns its result."""
    tool = _make_tool("t", fn=AsyncMock(return_value="done"))
    tc = _make_tc("t")
    agent = _make_agent(tool=tool)
    _, result, duration = await exec_tool(tc, agent)
    assert result == "done"
    assert duration >= 0


@pytest.mark.asyncio
async def test_exec_tool_on_before_tool_injection_skips_real_tool():
    """on_before_tool returning a string skips execution and uses the injected value."""
    executed = []

    async def real_fn() -> str:
        executed.append(True)
        return "real"

    tool = _make_tool("t", fn=real_fn)

    class InjectHook(AgentHooks):
        async def on_before_tool(self, agent, name, args):
            """Inject a fixed result."""
            return "injected"

    tc = _make_tc("t")
    agent = _make_agent(tool=tool, hooks=InjectHook())
    _, result, _ = await exec_tool(tc, agent)
    assert result == "injected"
    assert not executed


@pytest.mark.asyncio
async def test_exec_tool_on_after_tool_can_replace_result():
    """on_after_tool returning a string replaces the real tool result."""
    tool = _make_tool("t", fn=AsyncMock(return_value="original"))

    class ReplaceHook(AgentHooks):
        async def on_after_tool(self, agent, name, args, result):
            """Return a replacement result."""
            return "replaced"

    tc = _make_tc("t")
    agent = _make_agent(tool=tool, hooks=ReplaceHook())
    _, result, _ = await exec_tool(tc, agent)
    assert result == "replaced"
