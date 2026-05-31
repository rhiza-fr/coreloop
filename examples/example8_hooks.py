"""Example showing lifecycle hooks — tool logging, timing, and usage tracking.

run this with

uv run examples/example8_hooks.py what files are in this project?
"""

import asyncio
import logging
import sys
import time
from typing import Any

from minimal_agent import Agent, AgentHooks, Message, Usage
from minimal_agent._logging import setup_logging


class LoggingHook(AgentHooks):
    """Logs at all lifecycle events"""

    def __init__(self) -> None:
        self._log = logging.getLogger("minimal_agent.hooks")

    async def on_before_agent(self, agent: Agent) -> None:
        self._log.debug("on_before_agent: model=%s", agent.model)

    async def on_after_agent(self, agent: Agent) -> None:
        self._log.debug("on_after_agent: %d messages in history", len(agent.messages))

    async def on_before_turn(self, agent: Agent) -> None:
        self._log.info("on_before_turn")

    async def on_after_turn(self, agent: Agent) -> None:
        self._log.info("on_after_turn")

    async def on_before_llm(self, agent: Agent) -> Message | None:
        self._log.debug("on_before_llm: sending %d messages", len(agent.messages))
        return None  # None = proceed with the LLM call as normal

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        tool_names = [tc.function.name for tc in (message.tool_calls or [])]
        if tool_names:
            self._log.debug("on_after_llm: tool_calls=%s", tool_names)
        else:
            preview = (message.content or "")[:80].replace("\n", " ")
            self._log.debug("on_after_llm: content=%r usage=%s", preview, message.usage)
        return None  # None = keep the LLM response unchanged

    async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
        self._log.debug("on_before_tool: %s(%s)", name, args)
        return None  # None = execute the tool normally

    async def on_after_tool(
        self, agent: Agent, name: str, args: dict[str, Any], result: str
    ) -> str | None:
        self._log.debug("on_after_tool: %s -> %s", name, result[:120].replace("\n", " "))
        return None  # None = keep the tool result unchanged


class TimingHook(AgentHooks):
    def __init__(self) -> None:
        self._start = 0.0

    async def on_before_agent(self, agent: Agent) -> None:
        self._start = time.perf_counter()

    async def on_after_agent(self, agent: Agent) -> None:
        print(f"finished in {time.perf_counter() - self._start:.2f}s")


class UsageHook(AgentHooks):
    """Keeps track of LLM Call usage"""

    def __init__(self) -> None:
        self._usage = Usage()
        self._calls = 0

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        self._calls += 1
        if message.usage:
            self._usage.prompt_tokens += message.usage.prompt_tokens
            self._usage.completion_tokens += message.usage.completion_tokens
            self._usage.total_tokens += message.usage.total_tokens
        return None  # None = don't modify the LLM response — we're just observing usage

    async def on_after_agent(self, agent: Agent) -> None:
        print(
            f"LLM calls: {self._calls} | "
            f"prompt={self._usage.prompt_tokens} "
            f"completion={self._usage.completion_tokens} "
            f"total={self._usage.total_tokens} tokens"
        )


class DemoHooks(LoggingHook, TimingHook, UsageHook):
    """Compose multiple hooks via explicit delegation.

    When two base classes both define the same hook method, only the first one
    runs unless you override it here and call both explicitly. The overrides
    below are the ones where that collision occurs. Methods unique to a single
    base (on_before_tool, on_before_turn, on_after_turn) are inherited as-is.
    """

    def __init__(self) -> None:
        LoggingHook.__init__(self)
        TimingHook.__init__(self)
        UsageHook.__init__(self)

    async def on_before_agent(self, agent: Agent) -> None:
        await LoggingHook.on_before_agent(self, agent)
        await TimingHook.on_before_agent(self, agent)

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        await LoggingHook.on_after_llm(self, agent, message)
        return await UsageHook.on_after_llm(self, agent, message)

    async def on_after_agent(self, agent: Agent) -> None:
        await LoggingHook.on_after_agent(self, agent)
        await TimingHook.on_after_agent(self, agent)
        await UsageHook.on_after_agent(self, agent)


async def main(prompt: str) -> None:
    setup_logging(logging.DEBUG)
    logging.getLogger("minimal_agent._execution").setLevel(
        logging.WARNING
    )  # suppress verbose internal execution logs so only hook output is visible
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read", "grep"],
        root=".",
        hooks=DemoHooks(),
    )

    async for msg in agent.run([Message(role="user", content=prompt)]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example8_hooks.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
