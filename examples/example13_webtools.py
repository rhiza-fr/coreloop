"""Example showing web tools — web_search and web_fetch.

Web tools are optional and require the [web] extra:

    pip install minimal-agent[web]
    or ... uv sync --all-extras (if you are in this src)

They also require a running SearXNG instance for search. Set its URL via:

    export SEARXNG_URL=http://localhost:8080

or pass it directly to make_web_tools(). web_fetch works without SearXNG.

Unlike file tools (passed as strings like "read"), web tools are constructed
with make_web_tools() and passed as ToolInfo objects. This lets you configure
the SearXNG URL at construction time rather than globally.

run this with

uv run examples/example13_webtools.py
"""

import asyncio
import os

from minimal_agent import Agent, Message, make_web_tools


async def main() -> None:
    # make_web_tools() reads SEARXNG_URL from the environment if not passed.
    # Omit searxng_url entirely if the env var is set.
    searxng_url = os.environ.get(
        "SEARXNG_URL", "http://search.lan"
    )  # Common LAN hostname for SearXNG
    tools = make_web_tools(searxng_url=searxng_url)  # Returns ToolInfo objects, not strings

    agent = Agent(
        model="qwen3.5:9b",
        tools=tools,
        system="You are a research assistant. Cite URLs when you use them.",
    )

    prompts = [
        # web_search: find current information
        "What is the latest stable version of Python?",
        # web_fetch: read a specific page the agent already knows
        "Fetch https://python.org and tell me what Python versions are highlighted.",
    ]

    for prompt in prompts:
        print(f"Q: {prompt}")
        async for msg in agent.run([Message(role="user", content=prompt)]):
            # Skip streaming partials — only print complete assistant replies
            if not msg.partial and msg.role == "assistant" and msg.content:
                print(f"A: {msg.content}")
        agent.reset()  # Clear conversation history so each query is independent
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
