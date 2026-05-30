"""Tool execution — argument validation, timeout handling, and error formatting."""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from .hooks import _safe_hook
from .registry import ToolInfo
from .types import ToolCall

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


async def exec_tool(tc: ToolCall, agent: "Agent") -> tuple[ToolCall, str, float]:
    """Execute a single tool call and return (call, result, duration)."""
    name = tc.function.name
    try:
        args: dict[str, Any] = (
            json.loads(tc.function.arguments) if tc.function.arguments else {}
        )
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse arguments for tool '%s': %s", name, exc)
        return tc, f"Error: failed to parse arguments for '{name}': {exc}", 0.0

    injected = await _safe_hook(agent.hooks, "on_before_tool", agent, name, args)
    if injected is not None:
        logger.debug("Tool '%s' result injected by on_before_tool hook", name)
        await _safe_hook(agent.hooks, "on_after_tool", agent, name, args, injected)
        return tc, injected, 0.0

    info = agent._resolve_tool(name)
    if info is None:
        logger.warning("Unknown tool requested: '%s'", name)
        result = (
            f"Error: unknown tool '{name}'. "
            f"Available tools: {', '.join(t.name for t in agent._all_tools())}"
        )
        await _safe_hook(agent.hooks, "on_after_tool", agent, name, args, result)
        return tc, result, 0.0

    logger.info("Tool call: %s(%s)", name, tc.function.arguments or "")
    t0 = time.perf_counter()
    result = await run_tool(info, args, agent.timeout)
    duration = time.perf_counter() - t0
    logger.info("Tool '%s' completed in %.2fs", name, duration)

    await _safe_hook(agent.hooks, "on_after_tool", agent, name, args, result)
    return tc, result, duration


async def run_tool(info: ToolInfo, args: dict[str, Any], timeout: float) -> str:
    """Validate arguments and execute a tool with timeout."""
    allowed = set(info.parameters.get("properties", {}).keys())
    required = set(info.parameters.get("required", []))
    missing = required - set(args.keys())
    unknown = set(args.keys()) - allowed
    if missing:
        return (
            f"Error: tool '{info.name}' missing required arguments: "
            f"{', '.join(sorted(missing))}"
        )
    if unknown:
        return (
            f"Error: tool '{info.name}' received unexpected arguments: "
            f"{', '.join(sorted(unknown))}"
        )

    try:
        async with asyncio.timeout(timeout):
            result = await info.fn(**args)
        return str(result)
    except asyncio.TimeoutError:
        logger.error("Tool '%s' timed out after %.1fs", info.name, timeout)
        return f"Error: tool '{info.name}' timed out after {timeout}s"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Tool '%s' raised %s: %s", info.name, type(exc).__name__, exc)
        return f"Error in tool '{info.name}': {exc}"
