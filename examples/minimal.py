"""Minimal example — the simplest possible agent run."""

import asyncio

from minimal_agent import Agent, Message


async def main() -> None:
    agent = Agent(
        model="qwen3.5:9b",
        provider="ollama",
    )

    async for msg in agent.run([Message(role="user", content="Say hello in one sentence.")]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
