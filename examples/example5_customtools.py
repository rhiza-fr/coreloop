"""Example showing custom tools via the @tool decorator.

The @tool decorator registers an async function as a tool the agent can call.
Parameter schemas are inferred from type annotations; the docstring becomes the
tool description. Tools registered this way are visible to all Agent instances.

run this with

uv run examples/example5_customtools.py
"""

import asyncio
import datetime
import json
import math

from minimal_agent import Agent, Message, tool


@tool
async def get_current_time() -> str:
    """Return the current local time as HH:MM:SS."""
    return datetime.datetime.now().strftime("%H:%M:%S")


@tool
async def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression and return the result.

    Supports standard arithmetic and math module functions (sqrt, sin, cos, …).
    """
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
async def reverse_string(text: str) -> str:
    """Return the input string reversed."""
    return text[::-1]


async def main() -> None:
    agent = Agent(
        model="qwen3.5:9b",
        tools=["get_current_time", "calculate", "reverse_string"],
        system="Use the provided tools to answer questions. Be concise.",
    )

    prompts = [
        "What time is it right now?",
        "What is the square root of 144 plus 7 squared?",
        'Reverse the string "Hello, world!"',
    ]

    for prompt in prompts:
        print(f"Q: {prompt}")
        async for msg in agent.run([Message(role="user", content=prompt)]):
            if not msg.partial and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    print(f"  → {tc.function.name}({args_str})")
            if not msg.partial and msg.role == "assistant" and msg.content:
                print(f"A: {msg.content}")
        agent.reset()
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
