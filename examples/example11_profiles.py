"""Example showing profiles -- named configurations from minimal-agent.toml.

Profiles let you define agent configurations in minimal-agent.toml and load them
by name at runtime. Every profile inherits from [profiles.default]; named
profiles override only the keys they specify. String values support
{{ENV_VAR}} interpolation.

Example minimal-agent.toml:

    [profiles.default]
    model = "qwen3.5:9b"
    base_url = "http://localhost:11434/v1"

    [profiles.openai]
    base_url = "https://api.openai.com/v1"
    api_key = "{{OPENAI_API_KEY}}"
    model = "gpt-4o-mini"

Two ways to load a profile:

  1. Agent.from_profile("openai")          -- convenience classmethod
  2. resolve_profile("openai")             -- returns AgentConfig for inspection
     Agent.from_config(cfg)

run this with

uv run examples/example11_profiles.py
"""

import asyncio

from minimal_agent import Agent, Message
from minimal_agent.profiles import resolve_profile


async def run(agent: Agent, prompt: str) -> None:
    async for msg in agent.run([Message(role="user", content=prompt)]):
        # Skip streaming partials -- only print complete assistant replies
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


async def main() -> None:
    # --- Option 1: convenience classmethod ---
    print("=== Agent.from_profile('default') ===")
    agent = Agent.from_profile("default")
    await run(agent, "What is your model name?")

    # --- Option 2: inspect the config before constructing ---
    print("\n=== resolve_profile then from_config ===")
    cfg = resolve_profile("default")
    print(f"  model={cfg.model!r}  base_url={cfg.base_url!r}")
    agent = Agent.from_config(cfg)
    await run(agent, "What is 3 + 3?")

    # --- Switching profiles at runtime ---
    # Uncomment and set OPENAI_API_KEY to try the openai profile:
    #
    # print("\n=== openai profile ===")
    # agent = Agent.from_profile("openai")
    # await run(agent, "What is your model name?")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Suppress traceback on Ctrl+C -- asyncio.run re-raises differently
