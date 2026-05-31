"""Example showing the raw message stream — every completed message pretty-printed.

Useful for understanding what the agent loop actually produces: assistant
messages with tool_calls, tool result messages, usage, timing, etc.

run this with

uv run examples/example7_raw_messages.py what files are in this project?
"""

import asyncio
import sys

from rich.pretty import pprint

from minimal_agent import Agent, Message


async def main(prompt: str) -> None:
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read", "grep"],
        root=".",
    )

    async for msg in agent.run([Message(role="user", content=prompt)]):
        if not msg.partial:  # skip streaming partials — only show completed messages
            pprint(
                msg.model_dump(exclude_none=True)
            )  # exclude None values so the dump is compact and readable
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example7_raw_messages.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
