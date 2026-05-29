"""File tools example — agent with read, ls, edit, and search."""

import asyncio

from minimal_agent import Agent, Message, make_tools


async def main() -> None:
    # make_tools returns read, ls, edit, search scoped to allowed_root.
    # The agent cannot access files outside this directory.
    tools = make_tools(allowed_root=".")

    agent = Agent(
        model="qwen3.5:9b",
        provider="ollama",
        tools=tools,
        system="You are a helpful assistant with access to the local filesystem.",
    )

    async for msg in agent.run([
        Message(role="user", content="What files are in the current directory?"),
    ]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
