"""Example showing multi-turn conversation — reusing message history across runs.

After agent.run() completes, agent.messages holds the full history: system
prompt, all assistant turns, and all tool results. Pass it as the starting
messages on the next run to continue the conversation.

run this with

uv run examples/example2_conversation.py
"""

import asyncio

from minimal_agent import Agent, Message


async def ask(agent: Agent, user_input: str) -> str:
    """Append a user message, run the agent, return the final assistant reply."""
    messages = agent.messages + [Message(role="user", content=user_input)]
    reply = ""
    async for msg in agent.run(messages):
        if not msg.partial and msg.role == "assistant" and msg.content:
            reply = msg.content
    return reply


async def main() -> None:
    agent = Agent(
        model="qwen3.5:9b",
        system="You are a helpful assistant. Keep answers brief.",
    )

    # Turn 1 — agent has no history yet
    reply = await ask(agent, "My name is Alice. Remember that.")
    print(f"Turn 1: {reply}")

    # Turn 2 — agent.messages now contains the first exchange
    reply = await ask(agent, "What is my name?")
    print(f"Turn 2: {reply}")

    # Turn 3 — full history carries forward
    reply = await ask(agent, "How many messages have we exchanged so far?")
    print(f"Turn 3: {reply}")

    print(f"\n{len(agent.messages)} messages in history")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
