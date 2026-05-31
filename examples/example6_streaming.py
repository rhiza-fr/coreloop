"""Example with streaming output and file tools.

Partial messages arrive with accumulated content, so we track how much
has already been printed and only write the new characters each time.

run this with

uv run examples/example6_streaming.py what files are in this project?
"""

import asyncio
import sys

from minimal_agent import Agent, Message


async def main(prompt: str) -> None:
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read", "grep"],  # add the "edit" tool if you wish
        root=".",
    )

    printed = 0
    async for msg in agent.run(messages=[Message(role="user", content=prompt)]):
        if msg.role == "assistant":
            if msg.content:
                print(msg.content[printed:], end="", flush=True)
                printed: int = len(msg.content)
            if not msg.partial:
                print()
                printed = 0
        elif msg.role == "tool":
            preview: str = (msg.content or "")[:120].replace("\n", " ")
            print(f"[{msg.name}] {preview}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example6_streaming.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
