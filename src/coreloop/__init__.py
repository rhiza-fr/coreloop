"""minimal-agent: a minimal LLM agent with tool-calling support."""

from .agent import Agent
from .config import AgentConfig
from .tools import make_tools
from .tools.bash import make_bash_tool
from .tools.grep import make_grep_tool
from .hooks import AgentHooks, MaxTurnsHook
from .tool_registry import ToolInfo, clear_registry, get_tool, list_tools, tool
from .web_tools import make_web_tools
from .types import FunctionCall, Message, ToolCall, Usage

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentHooks",
    "FunctionCall",
    "MaxTurnsHook",
    "Message",
    "ToolCall",
    "ToolInfo",
    "Usage",
    "clear_registry",
    "get_tool",
    "list_tools",
    "make_bash_tool",
    "make_grep_tool",
    "make_tools",
    "make_web_tools",
    "tool",
]

__version__ = "0.1.0"
