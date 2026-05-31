"""Example with built-in file tools (ls, read, grep).

run this with

uv run examples/example4_builtintools.py what is the name of this project?
"""

import asyncio
import json
import sys

from minimal_agent import Agent, Message


async def main(prompt: str) -> None:
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read", "grep"],  # edit, bash, web_search, web_fetch also available
        root=".",  # Required: sandboxes file tools to this directory
    )

    async for msg in agent.run([Message(role="user", content=prompt)]):
        if not msg.partial and msg.tool_calls:
            for tc in msg.tool_calls:
                # arguments may be None or "{}" — default to empty dict for safety
                args = json.loads(tc.function.arguments or "{}")
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  → {tc.function.name}({args_str})")
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example4_builtintools.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
