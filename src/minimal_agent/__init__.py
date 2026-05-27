"""minimal-agent: a minimal LLM agent with tool-calling support."""

from __future__ import annotations

from ._agent import Agent
from ._builtin_tools import make_tools
from ._tool import ToolInfo, clear_registry, tool
from ._types import FunctionCall, Message, ToolCall, Usage

__all__ = [
    "Agent",
    "FunctionCall",
    "Message",
    "ToolCall",
    "ToolInfo",
    "Usage",
    "clear_registry",
    "make_tools",
    "tool",
]

__version__ = "0.1.0"
