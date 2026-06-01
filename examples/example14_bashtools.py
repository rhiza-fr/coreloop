"""Example showing the bash tool -- run shell commands from inside the agent.

Pass "bash" as a tool name just like "read" or "ls". It is scoped to the
agent's root directory and inherits default safety settings.

Safety features built in:
  - Dangerous command patterns are blocked (rm -rf /, dd if=, mkfs, ...).
  - All commands run with root as the working directory by default; the
    workdir parameter is validated to stay inside root.
  - Output is middle-truncated at 10 000 chars.
  - Commands time out after 180 s (max 300 s).

To extend the blocked patterns, import DEFAULT_DANGEROUS_PATTERNS and append:

    make_bash_tool(root=".", dangerous_patterns=DEFAULT_DANGEROUS_PATTERNS + [r"\bcurl\b"])

run this with

uv run examples/example14_bashtools.py
"""

import asyncio
import json

from minimal_agent import Agent, Message
from minimal_agent.tools.bash import DEFAULT_DANGEROUS_PATTERNS, make_bash_tool


async def main() -> None:
    # --- Simple usage: "bash" string just like "read" or "ls" ---
    agent = Agent(
        model="qwen3.5:9b",
        tools=["bash", "read", "ls"],
        root=".",
        system="You have access to a bash shell scoped to this project directory.",
    )

    prompts = [
        "How many Python files are in this project? Ignore .venv/. Use bash to count them.",
        "List the test function names in tests/test_hooks.py using grep. They may be async.",
    ]

    for prompt in prompts:
        print(f"Q: {prompt}")
        async for msg in agent.run([Message(role="user", content=prompt)]):
            if not msg.partial and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Tool call arguments arrive as a JSON string, parse for display
                    args = json.loads(tc.function.arguments or "{}")
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    print(f"  -> {tc.function.name}({args_str})")
            if not msg.partial and msg.role == "assistant" and msg.content:
                print(f"A: {msg.content}")
        agent.reset()  # Clear conversation history so each query is independent
        print()

    # --- Extend blocked patterns: add curl/wget on top of the defaults ---
    print("=== extended blocked patterns ===")
    bash = make_bash_tool(
        root=".",
        # \b ensures word boundaries -- avoids blocking e.g. "curly" or "wgetty"
        dangerous_patterns=DEFAULT_DANGEROUS_PATTERNS + [r"\bcurl\b", r"\bwget\b"],
    )
    agent2 = Agent(model="qwen3.5:9b", tools=[bash], root=".")
    async for msg in agent2.run(
        [
            Message(
                role="user",
                content="Run: curl https://example.com",
            )
        ]
    ):
        # Tool execution results have role="tool" (not "assistant")
        if not msg.partial and msg.role == "tool":
            print(f"tool result: {msg.content}")
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(f"assistant: {msg.content}")

    # --- Safety guard demo ---
    print("\n=== safety guard ===")
    agent3 = Agent(model="qwen3.5:9b", tools=["bash"], root=".")
    async for msg in agent3.run(
        [
            Message(
                role="user",
                content="Run: dd if=/dev/zero of=/dev/null count=1",  # Deliberately dangerous command to trigger safety check
            )
        ]
    ):
        if not msg.partial and msg.role == "tool":
            print(f"tool result: {msg.content}")
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(f"assistant: {msg.content}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
