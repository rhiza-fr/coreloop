"""Example showing subagents — tools that themselves run an Agent.

A subagent is just an async tool that instantiates an Agent, drains run(),
and returns the final reply as a string. The parent sees it as an ordinary
tool result.

Parallelism comes for free: the agent loop runs all tool calls in a single
LLM turn concurrently (asyncio.gather). If the LLM calls `delegate` twice
in one response, both subagents run at the same time.

You can also dispatch subagents explicitly by gathering them inside a single
tool call — useful when you want to guarantee parallel execution regardless
of how the LLM structures its calls.

Two things to watch:
  - tool_timeout on the parent must be high enough for the subagent to finish.
  - Each subagent uses its own LLM cache, so repeated identical subtasks are free.

run this with

uv run examples/example12_subagents.py
"""

import asyncio

from minimal_agent import Agent, MaxTurnsHook, Message, tool


def _make_subagent() -> Agent:
    return Agent(
        model="qwen3.5:9b",
        system="Complete the given task concisely. One short paragraph max.",
        hooks=MaxTurnsHook(3),  # Cap subagent at 3 turns to prevent runaway loops
    )


# --- Pattern 1: single-task delegation ---


@tool
async def delegate(task: str) -> str:
    """Run a subtask with a dedicated agent and return its response."""
    sub = _make_subagent()
    result = ""
    async for msg in sub.run([Message(role="user", content=task)]):
        # Skip streaming partials — we only want the final complete response
        if not msg.partial and msg.role == "assistant" and msg.content:
            result = msg.content
    return result or "(no response)"  # Fallback if subagent produces zero non-partial messages


# --- Pattern 2: explicit parallel dispatch ---


@tool
async def delegate_parallel(tasks: list[str]) -> str:
    """Run multiple independent subtasks in parallel and return all results.

    Each task gets its own agent; all run concurrently via asyncio.gather.
    Results are returned as a numbered list.
    """

    async def run_one(task: str) -> str:
        sub = _make_subagent()
        result = ""
        async for msg in sub.run([Message(role="user", content=task)]):
            if not msg.partial and msg.role == "assistant" and msg.content:
                result = msg.content
        return result or "(no response)"

    # All subagents run concurrently via gather — no explicit threading needed
    results = await asyncio.gather(*[run_one(t) for t in tasks])
    return "\n\n".join(f"[{i + 1}] {r}" for i, r in enumerate(results))


async def main() -> None:
    # --- Pattern 1: LLM decides to call delegate (possibly multiple times) ---
    print("=== single delegate ===")
    agent = Agent(
        model="qwen3.5:9b",
        tools=["delegate"],
        tool_timeout=120.0,  # Subagents need extra time; default 60s may be too tight
        system="Use the delegate tool to answer questions you cannot answer alone.",
    )
    async for msg in agent.run(
        [
            Message(
                role="user",
                content="Use delegate to ask: what is the capital of France?",
            )
        ]
    ):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)

    # --- Pattern 2: explicit parallel dispatch in one tool call ---
    print("\n=== parallel delegate ===")
    agent = Agent(
        model="qwen3.5:9b",
        tools=["delegate_parallel"],
        tool_timeout=120.0,  # Subagents need extra time; default 60s may be too tight
        system="Use delegate_parallel to handle multiple subtasks at once.",
    )
    async for msg in agent.run(
        [
            Message(
                role="user",
                content=(
                    "Use delegate_parallel to answer these three questions simultaneously: "
                    "(1) What is photosynthesis? "
                    "(2) What is the Pythagorean theorem? "
                    "(3) What is the speed of light?"
                ),
            )
        ]
    ):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Suppress traceback on Ctrl+C
