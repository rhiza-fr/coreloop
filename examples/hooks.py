"""Example AgentHooks implementations.

AgentHooks is a base class with async methods that fire at key points in the
agent loop.  Subclass it and override only the methods you need — the base
implementations are all no-ops (except on_before_tool, which returns None to
signal "proceed normally").

Hook firing order for a single agent.run() call:

    on_before_agent(agent)                    # once, before any LLM call

    for each turn:
        on_before_turn(agent)             # brackets the full LLM + tool cycle

        on_before_llm(agent) -> Message | None
            # Return a Message to inject as the response (LLM call is skipped).
            # Return None to call the LLM normally.
        <LLM streams response>
        on_after_llm(agent, message) -> Message | None
            # Return a Message to replace what goes into conversation history.
            # The streamed content has already been yielded — only history is affected.

        if the LLM requested tool calls:
            for each tool (in parallel):
                on_before_tool(agent, name, args)  # return str to inject result
                <tool executes>
                on_after_tool(agent, name, args, result)

        on_after_turn(agent)              # fires every turn, with or without tools

    on_after_agent(agent)                     # run finished cleanly (or stop() called)
    # on_after_agent is NOT called after abort()

All hook methods are called with _safe_hook, which catches and logs any
exception rather than letting it propagate into the agent loop.  This means
hook bugs cannot crash the agent, but they also cannot interrupt it — use
agent.stop() inside a hook to request a clean exit.
"""

import asyncio
import logging
import time
from typing import Any

from minimal_agent import Agent, AgentHooks, Message, Usage
from minimal_agent.tools import make_tools

# ── Single-responsibility hooks ───────────────────────────────────────────────
#
# Each class below does exactly one thing.  They are designed to be composed
# (see the Composition section at the bottom) rather than doing too much in one
# place.


class MaxTurnsHook(AgentHooks):
    """Stop the agent after N turns.

    A "turn" is one full LLM + optional tool cycle.  on_after_turn fires at
    the end of every turn regardless of whether tools were called, so this
    counts all turns including pure text responses.

    Use this as a safety valve to prevent runaway loops.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._turns = 0

    async def on_after_turn(self, agent: Agent) -> None:
        # on_after_turn fires at the end of every turn — after tools if the
        # LLM called any, or immediately after the LLM response if not.
        # Calling agent.stop() here lets the current turn complete before
        # the loop exits.  Use on_before_turn to stop before the LLM is called.
        self._turns += 1
        if self._turns >= self._n:
            agent.stop()


class ToolLoggerHook(AgentHooks):
    """Print every tool call and its result to stdout.

    on_before_tool can either return None (proceed normally) or return a string
    to use as the tool result, skipping the actual tool execution entirely.
    This hook only observes, so it always returns None.
    """

    async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
        print(f"→ {name}({args})")
        return None  # None means "execute the tool normally"

    async def on_after_tool(self, agent: Agent, name: str, args: dict[str, Any], result: str) -> None:
        # result is always a string — tools that raise exceptions have their
        # error message captured into result rather than propagating.
        print(f"← {name}: {result[:120]}")


class TimingHook(AgentHooks):
    """Measure and report total wall-clock time for a run.

    on_before_agent fires exactly once at the start of run(), making it the
    ideal place to start a timer — no guard needed, unlike on_before_llm which
    fires once per LLM turn.
    """

    def __init__(self) -> None:
        self._start = 0.0

    async def on_before_agent(self, agent: Agent) -> None:
        self._start = time.perf_counter()

    async def on_after_agent(self, agent: Agent) -> None:
        # on_after_agent is the natural place for teardown and summary output.
        # It fires once, after the loop exits cleanly.  It is NOT called after
        # agent.abort() — use that distinction if you need to differentiate
        # clean exits from forced cancellations.
        elapsed = time.perf_counter() - self._start
        print(f"finished in {elapsed:.2f}s")


class LoggingHook(AgentHooks):
    """Log every hook invocation using Python's logging module.

    Attach to an agent to get a full trace of the lifecycle without modifying
    any behaviour — all methods return None / the base no-op.

    Usage::

        import logging
        logging.basicConfig(level=logging.DEBUG)
        agent = Agent(..., hooks=LoggingHook())
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        # Default to the minimal_agent namespace so setup_logging() covers it.
        self._log = logger or logging.getLogger("minimal_agent.hooks")

    async def on_before_agent(self, agent: Agent) -> None:
        self._log.debug("on_before_agent: model=%s provider=%s", agent.model, agent.provider)

    async def on_after_agent(self, agent: Agent) -> None:
        self._log.debug("on_after_agent: %d messages in history", len(agent.conversation))

    async def on_before_turn(self, agent: Agent) -> None:
        self._log.debug("on_before_turn")

    async def on_after_turn(self, agent: Agent) -> None:
        self._log.debug("on_after_turn")

    async def on_before_llm(self, agent: Agent) -> Message | None:
        self._log.debug("on_before_llm: sending %d messages", len(agent.conversation))
        return None

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        tool_names = [tc.function.name for tc in (message.tool_calls or [])]
        if tool_names:
            self._log.info("on_after_llm: tool_calls=%s", tool_names)
        else:
            preview = (message.content or "")[:80].replace("\n", " ")
            self._log.info("on_after_llm: content=%r usage=%s", preview, message.usage)
        return None

    async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
        self._log.info("on_before_tool: %s(%s)", name, args)
        return None

    async def on_after_tool(self, agent: Agent, name: str, args: dict[str, Any], result: str) -> None:
        preview = result[:120].replace("\n", " ")
        self._log.info("on_after_tool: %s -> %s", name, preview)


class UsageHook(AgentHooks):
    """Accumulate token usage across all LLM calls and print a summary at the end.

    on_after_llm receives the fully assembled Message for each LLM turn (not
    streaming partials).  If the provider supports usage reporting, the message
    will have a .usage field with prompt/completion/total token counts.

    This hook is useful for cost estimation and debugging unexpectedly large
    context windows.
    """

    def __init__(self) -> None:
        self._usage = Usage()
        self._calls = 0

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        self._calls += 1
        if message.usage:
            # Accumulate incrementally so the running totals are always current,
            # even if on_after_agent is never reached (e.g. after abort()).
            self._usage.prompt_tokens += message.usage.prompt_tokens
            self._usage.completion_tokens += message.usage.completion_tokens
            self._usage.total_tokens += message.usage.total_tokens
        return None

    async def on_after_agent(self, agent: Agent) -> None:
        print(
            f"LLM calls: {self._calls} | "
            f"prompt={self._usage.prompt_tokens} "
            f"completion={self._usage.completion_tokens} "
            f"total={self._usage.total_tokens} tokens"
        )


# ── Composition patterns ──────────────────────────────────────────────────────
#
# agent.hooks is a single slot, so combining behaviours requires one of the
# patterns below.  Each has different tradeoffs.
#
#
# Pattern 1 — Multiple inheritance (simple, but method resolution can surprise)
#
#   class MyHooks(MaxTurnsHook, ToolLoggerHook):
#       def __init__(self):
#           MaxTurnsHook.__init__(self, 10)
#           # ToolLoggerHook has no __init__, nothing to call
#
#   This works well when the hooks touch *different* methods.  If two base
#   classes both override the same method (e.g. both define on_after_agent),
#   Python's MRO only calls one of them unless you use super() chains or
#   explicit delegation (see Pattern 2).
#
#
# Pattern 2 — Explicit delegation (verbose but unambiguous)
#
#   class MyHooks(TimingHook, UsageHook):
#       def __init__(self):
#           TimingHook.__init__(self)
#           UsageHook.__init__(self)
#
#       async def on_before_agent(self, agent):
#           await TimingHook.on_before_agent(self, agent)
#           # UsageHook has no on_before_agent, nothing to call
#
#       async def on_before_llm(self, agent):
#           await TimingHook.on_before_llm(self, agent)
#           # UsageHook has no on_before_llm, nothing to call
#
#       async def on_after_llm(self, agent, message) -> Message | None:
#           return await UsageHook.on_after_llm(self, agent, message)
#
#       # Neither base class defines on_before_turn / on_after_turn, so no
#       # delegation needed — the AgentHooks no-op base is inherited directly.
#
#       async def on_after_agent(self, agent):
#           await TimingHook.on_after_agent(self, agent)
#           await UsageHook.on_after_agent(self, agent)
#
#   Use this when two base classes define the same hook and you need both to
#   run.  The call order is explicit and easy to reason about.
#
#
# Pattern 3 — Fan-out wrapper (most flexible, highest boilerplate)
#
#   class Multi(AgentHooks):
#       def __init__(self, *hooks: AgentHooks) -> None:
#           self._hooks = hooks
#
#       async def on_before_agent(self, agent):
#           for h in self._hooks:
#               await h.on_before_agent(agent)
#
#       async def on_before_turn(self, agent):
#           for h in self._hooks:
#               await h.on_before_turn(agent)
#
#       async def on_before_llm(self, agent) -> Message | None:
#           for h in self._hooks:
#               result = await h.on_before_llm(agent)
#               if result is not None:
#                   return result  # first injector wins
#           return None
#
#       async def on_after_llm(self, agent, message) -> Message | None:
#           for h in self._hooks:
#               replacement = await h.on_after_llm(agent, message)
#               if replacement is not None:
#                   message = replacement  # pass modified message down the chain
#           return message
#
#       async def on_before_tool(self, agent, name, args):
#           for h in self._hooks:
#               result = await h.on_before_tool(agent, name, args)
#               if result is not None:
#                   return result  # first injector wins
#           return None
#
#       async def on_after_tool(self, agent, name, args, result):
#           for h in self._hooks:
#               await h.on_after_tool(agent, name, args, result)
#
#       async def on_after_turn(self, agent):
#           for h in self._hooks:
#               await h.on_after_turn(agent)
#
#       async def on_after_agent(self, agent):
#           for h in self._hooks:
#               await h.on_after_agent(agent)
#
#   agent = Agent(..., hooks=Multi(MaxTurnsHook(5), ToolLoggerHook(), TimingHook()))
#
#   This is the most composable pattern — hooks are assembled at construction
#   time from independent objects with no shared state.  The downside is that
#   you must forward every method explicitly.


async def main() -> None:
    # Demonstrate Pattern 2: LoggingHook + TimingHook + UsageHook combined with
    # explicit delegation, since all three define on_after_agent or on_before_llm.
    from minimal_agent._logging import setup_logging

    setup_logging(logging.DEBUG)

    class DemoHooks(LoggingHook, TimingHook, UsageHook):
        def __init__(self) -> None:
            LoggingHook.__init__(self)
            TimingHook.__init__(self)
            UsageHook.__init__(self)

        async def on_before_agent(self, agent: Agent) -> None:
            await LoggingHook.on_before_agent(self, agent)
            await TimingHook.on_before_agent(self, agent)

        async def on_before_turn(self, agent: Agent) -> None:
            await LoggingHook.on_before_turn(self, agent)

        async def on_after_turn(self, agent: Agent) -> None:
            await LoggingHook.on_after_turn(self, agent)

        async def on_before_llm(self, agent: Agent) -> Message | None:
            return await LoggingHook.on_before_llm(self, agent)

        async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
            await LoggingHook.on_after_llm(self, agent, message)
            return await UsageHook.on_after_llm(self, agent, message)

        async def on_before_tool(self, agent: Agent, name: str, args: dict) -> str | None:
            return await LoggingHook.on_before_tool(self, agent, name, args)

        async def on_after_tool(self, agent: Agent, name: str, args: dict, result: str) -> None:
            await LoggingHook.on_after_tool(self, agent, name, args, result)

        async def on_after_agent(self, agent: Agent) -> None:
            await LoggingHook.on_after_agent(self, agent)
            await TimingHook.on_after_agent(self, agent)
            await UsageHook.on_after_agent(self, agent)

    agent = Agent(
        model="qwen3.5:9b",
        provider="ollama",
        tools=make_tools(),
        hooks=DemoHooks(),
    )
    async for msg in agent.run(
        [Message(role="user", content="What is the name of this project?")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
