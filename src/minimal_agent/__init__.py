"""minimal-agent: a minimal LLM agent with tool-calling support."""

from .agent import Agent
from ._builtin_tools import make_tools
from ._cache import make_cache
from .hooks import AgentHooks
from .tool import ToolInfo, clear_registry, tool
from ._web_tools import make_web_tools
from .types import FunctionCall, Message, ToolCall, Usage

__all__ = [
    "Agent",
    "AgentHooks",
    "FunctionCall",
    "Message",
    "ToolCall",
    "ToolInfo",
    "Usage",
    "clear_registry",
    "make_cache",
    "make_tools",
    "make_web_tools",
    "tool",
]

__version__ = "0.1.0"
