"""AgentConfig -- a portable, serialisable bundle of Agent constructor parameters.

Use with Agent.from_config(), or use dataclasses.replace() to derive variants:

    from dataclasses import replace
    from coreloop import AgentConfig

    base = AgentConfig(model="qwen3:8b")
    fast = replace(base, model="qwen3:0.6b", timeout=10.0)
    agent = Agent.from_config(fast)

Hooks are intentionally excluded -- they are stateful runtime objects, not config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """Portable, serialisable bundle of Agent constructor parameters."""
    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    system: str | None = None
    tools: list[str] = field(default_factory=list)
    root: str | None = None
    http_request_timeout: float = 300.0
    tool_timeout: float = 360.0
    llm_timeout: float = 300.0
    llm_extra_body: dict[str, Any] | None = None
    cache_dir: str | None = str(Path.home() / ".cache" / "coreloop-llm-cache")
