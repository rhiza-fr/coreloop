"""Minimal example -- the simplest possible agent run.

run this with

uv run examples/example1_minimal.py what is your name?
"""

import asyncio
import sys

from minimal_agent import Agent, Message


async def main(prompt: str) -> None:
    agent = Agent(
        model="qwen3.5:9b",
        # No base_url -> defaults to http://localhost:11434/v1 (Ollama)
    )

    async for msg in agent.run([Message(role="user", content=prompt)]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            # Only print final messages -- skip streaming partials and tool results
            print(msg.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_minimal.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
