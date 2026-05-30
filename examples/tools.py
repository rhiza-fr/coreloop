"""File tools example — agent with read, ls, edit, and search."""

import asyncio

from minimal_agent import Agent, Message


async def main() -> None:
    # Name the built-in tools you want; they're scoped to ``root``.
    # The agent cannot access files outside this directory.
    agent = Agent(
        model="qwen3.5:9b",
        provider="ollama",
        tools=["read", "ls", "edit", "search"],
        root=".",
        system="You are a helpful assistant with access to the local filesystem.",
    )

    async for msg in agent.run([
        Message(role="user", content="What files are in the current directory?"),
    ]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
