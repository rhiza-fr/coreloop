"""The Agent — orchestrates the LLM loop with tool execution."""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, AsyncIterator

from ._cache import make_cache
from ._client import stream_chat
from ._config import resolve_provider
from ._execution import exec_tool
from .hooks import AgentHooks, _safe_hook
from .registry import ToolInfo, get_tool
from .types import Message, ToolCall, Usage, _dump_messages

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "minimal-agent"

_FILE_TOOL_NAMES = frozenset({"read", "ls", "edit", "search"})
_WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch"})


def _resolve_tools(
    tools: "Sequence[str | ToolInfo]", root: str | Path | None
) -> dict[str, ToolInfo]:
    """Resolve a mixed list of tool names and ``ToolInfo`` objects.

    Names resolve, in order, to built-in file tools (scoped to *root*), built-in
    web tools, then globally registered ``@tool`` functions. ``ToolInfo`` objects
    (including ``@tool``-decorated functions) are used as-is. Later entries win on
    name collisions. Raises ``ValueError`` for an unknown name.
    """
    resolved: dict[str, ToolInfo] = {}
    file_tools: dict[str, ToolInfo] | None = None
    web_tools: dict[str, ToolInfo] | None = None

    for item in tools:
        if isinstance(item, ToolInfo):
            resolved[item.name] = item
            continue
        name = item
        if name in _FILE_TOOL_NAMES:
            if file_tools is None:
                from .tools import make_tools

                file_tools = {t.name: t for t in make_tools(root)}
            resolved[name] = file_tools[name]
        elif name in _WEB_TOOL_NAMES:
            if web_tools is None:
                from .web_tools import make_web_tools

                web_tools = {t.name: t for t in make_web_tools()}
            resolved[name] = web_tools[name]
        else:
            info = get_tool(name)
            if info is None:
                builtins = ", ".join(sorted(_FILE_TOOL_NAMES | _WEB_TOOL_NAMES))
                raise ValueError(
                    f"Unknown tool {name!r}. Built-ins: {builtins}; "
                    "or register one with @tool."
                )
            resolved[name] = info
    return resolved


class Agent:
    """A minimal agent that calls an LLM, executes tools, and loops.

    Usage::

        agent = Agent(
            model="gpt-4o-mini",
            provider="openai",
            system="You are a helpful assistant.",
            timeout=30,
        )

        async for msg in agent.run([Message(role="user", content="Hello!")]):
            print(msg)

    After ``run()`` completes (or is stopped), the full message history —
    including system prompt, assistant responses, and tool results — is
    available via ``agent.messages``.  You can copy it to a new agent to
    restart::

        # agent.messages contains every message the LLM saw
        new_agent = Agent(model="better-model", ...)
        async for msg in new_agent.run(agent.messages):
            ...

    The agent core has no built-in turn limit; to bound a run, attach a hook
    that calls ``agent.stop()`` (see ``MaxTurnsHook`` in the examples).
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        system: str | None = None,
        tools: Sequence[str | ToolInfo] | None = None,
        root: str | Path | None = None,
        timeout: float = 60.0,
        llm_timeout: float = 300.0,
        hooks: AgentHooks | None = None,
        extra_body: dict[str, Any] | None = None,
        cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        # Public — safe to read/write between runs
        self.model = model
        self.provider = provider
        self.system = system
        self.root = root
        self.timeout = timeout
        self.llm_timeout = llm_timeout
        self.hooks = hooks if hooks is not None else AgentHooks()
        self.extra_body = extra_body
        self._cache = make_cache(cache_dir) if cache_dir is not None else None

        self._provider_config = resolve_provider(provider)
        self._messages: list[Message] = []

        # Resolve names ('read', 'web_search', a registered @tool) and ToolInfo
        # objects into this agent's tool set. The agent only has the tools listed
        # here — there is no implicit inclusion of the global registry.
        self._tools: dict[str, ToolInfo] = (
            _resolve_tools(tools, root) if tools else {}
        )

        self._stop_event = asyncio.Event()
        self._current_task: asyncio.Task[None] | None = None
        self._aborted = False
        # Last message produced by the current LLM turn; set in _stream_llm_response.
        self._llm_last_chunk: Message | None = None

    # ── public API ────────────────────────────────────────────────

    @property
    def messages(self) -> list[Message]:
        """The message history from the last ``run()`` call (shallow copy)."""
        return list(self._messages)

    def reset(self) -> None:
        """Clear message history and reset the stop flag."""
        self._messages.clear()
        self._stop_event.clear()

    def stop(self) -> None:
        """Signal the agent to finish the current turn and stop cleanly.

        Safe to call from inside a tool or hook — sets the stop flag without
        cancelling the task, so the current turn completes normally and
        ``on_after_agent`` is called before the loop exits.
        """
        self._stop_event.set()

    def abort(self) -> None:
        """Halt immediately, abandoning in-flight tools.

        Cancels the current task. ``on_after_agent`` is NOT called. Use ``stop()``
        for a clean exit.
        """
        self._aborted = True
        self._stop_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    @property
    def stopped(self) -> bool:
        """Whether stop() or abort() has been called."""
        return self._stop_event.is_set()

    async def run(
        self,
        messages: list[Message],
        *,
        usage: Usage | None = None,
    ) -> AsyncIterator[Message]:
        """Run the agent loop, yielding messages as they are produced.

        The loop:
          1. Sends the message history to the LLM (streaming).
          2. If the LLM returns tool calls, executes each (with timeout).
          3. Appends results and repeats.
          4. Stops when the LLM returns a non-tool-call response.

        Streaming notes:
          - Intermediate content delta messages have ``partial=True``.
          - The final assembled message for each LLM turn has ``partial=False``.
          - Tool result messages always have ``partial=False``.
        """
        self._stop_event.clear()
        self._aborted = False
        self._current_task = asyncio.current_task()
        self._messages = list(messages)

        if self.system and not (
            self._messages and self._messages[0].role == "system"
        ):
            self._messages.insert(0, Message(role="system", content=self.system))

        logger.debug("Agent.run starting: model=%s provider=%s", self.model, self.provider)
        await _safe_hook(self.hooks, "on_before_agent", self)
        try:
            while not self._stop_event.is_set():
                await _safe_hook(self.hooks, "on_before_turn", self)

                async for msg in self._stream_llm_response(usage):
                    yield msg

                if self._stop_event.is_set() or self._llm_last_chunk is None:
                    return

                assistant_msg = self._llm_last_chunk
                self._messages.append(assistant_msg)

                if not assistant_msg.tool_calls:
                    logger.debug("LLM finished without tool calls")
                    await _safe_hook(self.hooks, "on_after_turn", self)
                    return

                logger.debug(
                    "LLM requested %d tool call(s): %s",
                    len(assistant_msg.tool_calls),
                    [tc.function.name for tc in assistant_msg.tool_calls],
                )
                results = await asyncio.gather(
                    *[exec_tool(tc, self) for tc in assistant_msg.tool_calls]
                )

                async for msg in self._emit_tool_results(results):
                    yield msg

                await _safe_hook(self.hooks, "on_after_turn", self)

        except asyncio.CancelledError:
            if self._stop_event.is_set():
                pass  # abort() — swallow
            else:
                raise
        finally:
            if not self._aborted:
                await _safe_hook(self.hooks, "on_after_agent", self)
            self._current_task = None

    # ── tool registry ─────────────────────────────────────────────

    def _all_tools(self) -> list[ToolInfo]:
        return list(self._tools.values())

    def _resolve_tool(self, name: str) -> ToolInfo | None:
        return self._tools.get(name)

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        all_tools = self._all_tools()
        if not all_tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in all_tools
        ]

    # ── loop helpers ──────────────────────────────────────────────

    async def _emit_tool_results(
        self, results: list[tuple[ToolCall, str, float]]
    ) -> AsyncIterator[Message]:
        for tc, result_content, tool_duration in results:
            tool_msg = Message(
                role="tool",
                content=result_content,
                tool_call_id=tc.id,
                name=tc.function.name,
                duration=tool_duration,
            )
            self._messages.append(tool_msg)
            yield tool_msg

    async def _stream_llm_response(
        self, usage: Usage | None
    ) -> AsyncIterator[Message]:
        self._llm_last_chunk = None
        injected = await _safe_hook(self.hooks, "on_before_llm", self)
        if injected is not None:
            self._llm_last_chunk = injected
            yield injected
            replacement = await _safe_hook(self.hooks, "on_after_llm", self, injected)
            if replacement is not None:
                self._llm_last_chunk = replacement
            return
        try:
            llm_deadline = self.llm_timeout if self.llm_timeout is not None else self.timeout
            async with asyncio.timeout(llm_deadline):
                async for chunk in stream_chat(
                    base_url=self._provider_config.base_url,
                    api_key=self._provider_config.api_key,
                    model=self.model,
                    messages=_dump_messages(self._messages),
                    tools=self._tool_schemas(),
                    timeout=self.timeout,
                    extra_body=self.extra_body,
                    usage=usage,
                    cache=self._cache,
                ):
                    if self._stop_event.is_set():
                        return
                    self._llm_last_chunk = chunk
                    if chunk.partial:
                        yield chunk
                    else:
                        yield chunk.model_copy()
                        replacement = await _safe_hook(self.hooks, "on_after_llm", self, chunk)
                        if replacement is not None:
                            self._llm_last_chunk = replacement
        except asyncio.CancelledError:
            if self._stop_event.is_set():
                return
            raise
