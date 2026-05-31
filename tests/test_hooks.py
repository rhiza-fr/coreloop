"""Tests for AgentHooks firing order and injection/replacement contracts."""

from typing import Any

import pytest

from minimal_agent import Agent, AgentHooks, Message
from minimal_agent.types import ToolCall, FunctionCall
from minimal_agent.registry import ToolInfo


# -- Helpers --------------------------------------------------------------------


def _text_msg(content: str = "hello") -> Message:
    return Message(role="assistant", content=content, model="test")


def _tool_msg(name: str, args: str = "{}") -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name=name, arguments=args))],
        model="test",
    )


def _noop_tool(name: str) -> ToolInfo:
    async def fn() -> str:
        return f"{name} result"

    return ToolInfo(
        name=name, description="test", parameters={"type": "object", "properties": {}}, fn=fn
    )


class RecordingHook(AgentHooks):
    """Records every hook invocation name; subclass to override individual hooks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def on_before_agent(self, agent: Agent) -> None:
        self.calls.append("before_agent")

    async def on_after_agent(self, agent: Agent) -> None:
        self.calls.append("after_agent")

    async def on_before_turn(self, agent: Agent) -> None:
        self.calls.append("before_turn")

    async def on_after_turn(self, agent: Agent) -> None:
        self.calls.append("after_turn")

    async def on_before_llm(self, agent: Agent) -> Message | None:
        self.calls.append("before_llm")
        return None

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        self.calls.append("after_llm")
        return None

    async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
        self.calls.append(f"before_tool:{name}")
        return None

    async def on_after_tool(
        self, agent: Agent, name: str, args: dict[str, Any], result: str
    ) -> None:
        self.calls.append(f"after_tool:{name}")


class InjectingHook(RecordingHook):
    """RecordingHook that injects LLM responses via on_before_llm, skipping HTTP."""

    def __init__(self, responses: list[Message]) -> None:
        super().__init__()
        self._responses = iter(responses)

    async def on_before_llm(self, agent: Agent) -> Message | None:
        await super().on_before_llm(agent)  # records "before_llm"
        return next(self._responses, _text_msg("[end]"))


def _agent(hook: AgentHooks, tools: list[ToolInfo] | None = None) -> Agent:
    # ollama requires no API key, so Agent construction always succeeds.
    return Agent(model="test", hooks=hook, tools=tools)


# -- Firing-order tests ---------------------------------------------------------


@pytest.mark.asyncio
async def test_text_turn_hook_order():
    """Single text response: all hooks fire in correct order."""
    hook = InjectingHook([_text_msg("hi")])
    async for _ in _agent(hook).run([Message(role="user", content="hello")]):
        pass

    assert hook.calls == [
        "before_agent",
        "before_turn",
        "before_llm",
        "after_llm",
        "after_turn",
        "after_agent",
    ]


@pytest.mark.asyncio
async def test_tool_turn_hook_order():
    """Tool-calling turn followed by text: correct order across both turns."""
    hook = InjectingHook([_tool_msg("mytool"), _text_msg("done")])
    agent = _agent(hook, tools=[_noop_tool("mytool")])

    async for _ in agent.run([Message(role="user", content="go")]):
        pass

    assert hook.calls == [
        "before_agent",
        # turn 1: LLM requests tool
        "before_turn",
        "before_llm",
        "after_llm",
        "before_tool:mytool",
        "after_tool:mytool",
        "after_turn",
        # turn 2: LLM returns text
        "before_turn",
        "before_llm",
        "after_llm",
        "after_turn",
        "after_agent",
    ]


# -- Injection / replacement contracts -----------------------------------------


@pytest.mark.asyncio
async def test_on_before_llm_injection_triggers_after_llm():
    """Injected response from on_before_llm still passes through on_after_llm."""
    hook = InjectingHook([_text_msg("injected")])
    collected: list[Message] = []
    async for msg in _agent(hook).run([Message(role="user", content="hi")]):
        collected.append(msg)

    assert "before_llm" in hook.calls
    assert "after_llm" in hook.calls
    assert any(m.content == "injected" for m in collected)


@pytest.mark.asyncio
async def test_on_after_llm_replaces_history():
    """Returning a Message from on_after_llm replaces what's stored in conversation history."""
    replacement = _text_msg("replaced")

    class ReplacingHook(InjectingHook):
        async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
            await super().on_after_llm(agent, message)
            return replacement

    hook = ReplacingHook([_text_msg("original")])
    agent = _agent(hook)
    async for _ in agent.run([Message(role="user", content="hi")]):
        pass

    assistant_msgs = [m for m in agent.messages if m.role == "assistant"]
    assert any(m.content == "replaced" for m in assistant_msgs)
    assert not any(m.content == "original" for m in assistant_msgs)


@pytest.mark.asyncio
async def test_on_before_tool_injection_skips_real_tool():
    """Returning a str from on_before_tool skips execution; on_after_tool still fires."""
    executed: list[str] = []

    async def real_tool() -> str:
        executed.append("ran")
        return "real"

    tool = ToolInfo(
        name="t", description="", parameters={"type": "object", "properties": {}}, fn=real_tool
    )

    class InjectingToolHook(InjectingHook):
        async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
            await super().on_before_tool(agent, name, args)
            return "injected"

    hook = InjectingToolHook([_tool_msg("t"), _text_msg("done")])
    agent = _agent(hook, tools=[tool])
    async for _ in agent.run([Message(role="user", content="go")]):
        pass

    assert not executed
    assert "before_tool:t" in hook.calls
    assert "after_tool:t" in hook.calls
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert any(m.content == "injected" for m in tool_results)


@pytest.mark.asyncio
async def test_on_after_agent_not_called_after_abort():
    """on_after_agent does not fire when the agent is aborted."""

    class AbortOnLlmHook(RecordingHook):
        async def on_before_llm(self, agent: Agent) -> Message | None:
            await super().on_before_llm(agent)
            agent.abort()
            return None

    hook = AbortOnLlmHook()
    try:
        async for _ in _agent(hook).run([Message(role="user", content="hi")]):
            pass
    except Exception:
        pass

    assert "after_agent" not in hook.calls
