"""AgentHooks — lifecycle callbacks for the agent loop.

Hooks fire in this order during a single agent.run() call:

    on_before_agent(agent)

    for each turn:
        on_before_turn(agent)           # brackets the full LLM + tool cycle

        on_before_llm(agent) -> Message | None
            # Return a Message to inject as the response (LLM call is skipped).
            # Return None to call the LLM normally.
        <LLM streams response>
        on_after_llm(agent, message) -> Message | None
            # Return a Message to replace what gets appended to conversation
            # history.  Return None to use the message as-is.
            # Note: the streamed content has already been yielded to the
            # caller — the replacement only affects conversation history.

        if the LLM requested tool calls:
            for each tool (in parallel):
                on_before_tool(agent, name, args) -> str | None
                    # Return a str to inject as the result (tool is skipped).
                    # Return None to execute the tool normally.
                <tool executes>
                on_after_tool(agent, name, args, result)

        on_after_turn(agent)            # fires every turn, with or without tools

    on_after_agent(agent)
    # NOT called after agent.abort() — only after natural end or stop().

All hooks are called via _safe_hook, which catches and logs any exception
rather than propagating it.  Hook bugs cannot crash the agent.  To request
a clean exit from inside a hook, call agent.stop().
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent
    from .types import Message

logger = logging.getLogger(__name__)


class AgentHooks:
    async def on_before_agent(self, agent: Agent) -> None:
        """Called once at the start of agent.run(), before any LLM call."""

    async def on_after_agent(self, agent: Agent) -> None:
        """Called when the agent finishes cleanly (natural end or stop()).
        Not called after agent.abort()."""

    async def on_before_llm(self, agent: Agent) -> Message | None:
        """Called before each LLM API call.

        Return a Message to inject as the response (LLM call is skipped).
        Return None to proceed normally.
        """
        return None

    async def on_after_llm(self, agent: Agent, message: Message) -> Message | None:
        """Called after each LLM API call with the assembled response message.

        Return a Message to replace the message before it is appended to
        conversation history.  Return None to use the message as-is.
        """
        return None

    async def on_before_tool(self, agent: Agent, name: str, args: dict[str, Any]) -> str | None:
        """Called before a tool executes.

        Return a str to inject as the tool result (real tool is skipped).
        Return None to proceed normally.
        """
        return None

    async def on_after_tool(self, agent: Agent, name: str, args: dict[str, Any], result: str) -> None:
        """Called after a tool executes with its result."""

    async def on_before_turn(self, agent: Agent) -> None:
        """Called at the start of each turn, before the LLM call."""

    async def on_after_turn(self, agent: Agent) -> None:
        """Called at the end of each turn, after tools (if any). Fires every turn."""


async def _safe_hook(hooks: AgentHooks, method: str, *args: Any) -> Any:
    """Call a hook method, logging and swallowing any exceptions."""
    try:
        return await getattr(hooks, method)(*args)
    except Exception as exc:
        logger.warning("Hook %s raised %s: %s", method, type(exc).__name__, exc)
