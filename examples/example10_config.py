"""Example showing AgentConfig — portable, serialisable agent configuration.

AgentConfig is a plain dataclass that bundles all Agent constructor parameters
except hooks (which are stateful runtime objects). Use it to:

  - Derive variants with dataclasses.replace()
  - Build configs programmatically and pass them around
  - Instantiate agents with Agent.from_config()

run this with

uv run examples/example10_config.py
"""

import asyncio
from dataclasses import replace

from minimal_agent import Agent, AgentConfig, Message


# Base config — everything else derives from this.
base_config = AgentConfig(
    model="qwen3.5:9b",
    system="Answer in one short sentence.",
)

# Derive a variant with a smaller model and tighter timeouts.
fast_config = replace(
    base_config, model="qwen3:0.6b", llm_timeout=30.0
)  # replace() copies base_config, overriding only listed fields

# Derive a variant that adds file tools.
file_config = replace(
    base_config, tools=["read", "ls", "grep"], root="."
)  # non-tool inherits system prompt from base_config


async def run(cfg: AgentConfig, prompt: str) -> None:
    agent = Agent.from_config(
        cfg
    )  # factory method: unpacks the dataclass fields into Agent() constructor kwargs
    async for msg in agent.run([Message(role="user", content=prompt)]):
        if not msg.partial and msg.role == "assistant" and msg.content:
            print(msg.content)


async def main() -> None:
    print("=== base_config ===")
    await run(base_config, "What is 2 + 2?")

    print("\n=== fast_config (smaller model) ===")
    await run(fast_config, "What is 2 + 2?")

    print("\n=== file_config (with tools) ===")
    await run(file_config, "List the top-level files in this directory.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
