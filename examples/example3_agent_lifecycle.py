"""Example showing agent lifecycle control — MaxTurnsHook, stop(), and abort().

The agent loop has no built-in turn limit; you control it via:

  MaxTurnsHook(n)  — stop cleanly after n turns (one LLM call + its tools)
  agent.stop()     — request a clean exit after the current turn finishes
  agent.abort()    — cancel immediately (on_after_agent hook is NOT called)
  agent.reset()    — clear history so the next run starts fresh

MaxTurnsHook resets its counter at on_before_agent, so reusing one instance
across multiple run() calls gives each run its own fresh budget.

run this with

uv run examples/example3_agent_lifecycle.py
"""

import asyncio

from minimal_agent import Agent, AgentHooks, MaxTurnsHook, Message


class TurnPrinter(AgentHooks):
    async def on_before_turn(self, agent: Agent) -> None:
        print(f"  [turn {len(agent.messages)}] LLM call starting…")

    async def on_after_agent(self, agent: Agent) -> None:
        print(f"  agent finished — {len(agent.messages)} messages in history")


async def main() -> None:
    # --- MaxTurnsHook: cap at 2 turns ---
    print("=== MaxTurnsHook(2) ===")
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls"],
        root=".",
        hooks=MaxTurnsHook(2),
    )
    async for msg in agent.run([Message(role="user", content="List every file, then summarise the project.")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)

    # --- Combining hooks: MaxTurnsHook + TurnPrinter ---
    print("\n=== MaxTurnsHook(3) + TurnPrinter ===")

    class BoundedWithLogging(TurnPrinter, MaxTurnsHook):
        def __init__(self) -> None:
            MaxTurnsHook.__init__(self, 3)

        async def on_after_turn(self, agent: Agent) -> None:
            await MaxTurnsHook.on_after_turn(self, agent)

        async def on_before_agent(self, agent: Agent) -> None:
            await MaxTurnsHook.on_before_agent(self, agent)

    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read"],
        root=".",
        hooks=BoundedWithLogging(),
    )
    async for msg in agent.run([Message(role="user", content="What does pyproject.toml contain?")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)

    # --- agent.reset(): reuse the same agent for a fresh run ---
    print("\n=== agent.reset() reuse ===")
    agent = Agent(model="qwen3.5:9b", hooks=TurnPrinter())
    async for msg in agent.run([Message(role="user", content="What is 1 + 1?")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(f"run 1: {msg.content}")

    agent.reset()  # clears history; next run starts clean

    async for msg in agent.run([Message(role="user", content="What did I ask you just before?")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(f"run 2 (after reset): {msg.content}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
