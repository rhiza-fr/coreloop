"""Example with streaming output, reasoning, and file tools.

Partial messages arrive with accumulated content *and* accumulated reasoning
(for thinking models), so we keep a separate cursor for each and only write the
new characters each time.  Reasoning tokens are dimmed and prefixed with a
"thinking>" marker; the final answer prints normally.

run this with

uv run examples/example6_streaming.py what files are in this project?
"""

import asyncio
import sys

from coreloop import Agent, Message

_DIM = "\033[2m"
_RESET = "\033[0m"


async def main(prompt: str) -> None:
    agent = Agent(
        model="qwen3.5:9b",
        tools=["ls", "read", "grep"],  # add the "edit" tool if you wish
        root=".",
        llm_extra_body={"reasoning_effort": "medium"},  # ask the model to think
    )

    # Two cursors: how many chars of the current message's reasoning / content
    # we've already printed.  Both reset when the final (non-partial) message lands.
    reasoned = 0
    printed = 0
    in_reasoning = False  # whether we've opened the dim "thinking" block
    async for msg in agent.run(messages=[Message(role="user", content=prompt)]):
        if msg.role == "assistant":
            if msg.reasoning and len(msg.reasoning) > reasoned:
                if not in_reasoning:
                    print(f"{_DIM}thinking> ", end="", flush=True)  # open the thinking block
                    in_reasoning = True
                print(msg.reasoning[reasoned:], end="", flush=True)  # only the new reasoning
                reasoned = len(msg.reasoning)
            if msg.content and len(msg.content) > printed:
                if in_reasoning:
                    print(f"{_RESET}\n", end="", flush=True)  # close thinking, start the answer
                    in_reasoning = False
                print(msg.content[printed:], end="", flush=True)  # only the new content
                printed = len(msg.content)
            if not msg.partial:  # final message in the stream: reset cursors and newline
                if in_reasoning:  # reasoning-only turn (e.g. before a tool call)
                    print(_RESET, end="", flush=True)
                    in_reasoning = False
                print()
                reasoned = 0
                printed = 0
        elif msg.role == "tool":
            preview = (msg.content or "")[:120].replace("\n", " ")  # compact one-line result
            print(f"[{msg.name}] {preview}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example6_streaming.py <prompt>", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(main(" ".join(sys.argv[1:])))
    except KeyboardInterrupt:
        pass
