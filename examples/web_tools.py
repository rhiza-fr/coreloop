"""Web tools example — agent with web_search and web_fetch.

Requires the web extra:
    pip install minimal-agent[web]

Requires a running SearXNG instance. Set the URL via:
    export SEARXNG_URL=http://localhost:8080
or pass it directly to make_web_tools().
"""

import asyncio

from minimal_agent import Agent, Message, make_web_tools


async def main() -> None:
    # SEARXNG_URL can also be set in ~/.ma-config.toml under [defaults].
    searxng_url = "http://search.lan"

    tools = make_web_tools(searxng_url=searxng_url)

    agent = Agent(
        model="qwen3.5:9b",
        provider="ollama",
        tools=tools,
        system="You are a helpful research assistant with access to web search and fetch.",
    )

    async for msg in agent.run([
        Message(role="user", content="What is the latest version of Python?"),
    ]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
