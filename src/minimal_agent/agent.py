"""The Agent — orchestrates the LLM loop with tool execution."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

from ._cache import make_cache
from ._client import stream_chat
from ._config import resolve_provider
from .hooks import AgentHooks, _safe_hook
from .tool import ToolInfo, list_tools
from .types import Message, ToolCall, Usage

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "minimal-agent"


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

    After ``run()`` completes (or is stopped), the full conversation — including
    system prompt, assistant responses, and tool results — is available via
    ``agent.conversation``.  You can copy it to a new agent to restart::

        # agent.conversation contains every message the LLM saw
        new_agent = Agent(model="better-model", ...)
        async for msg in new_agent.run(agent.conversation):
            ...
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        system: str | None = None,
        tools: list[ToolInfo] | None = None,
        timeout: float = 60.0,
        hooks: AgentHooks | None = None,
        max_messages: int = 0,
        extra_body: dict[str, Any] | None = None,
        cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        # Public — safe to read/write between runs
        self.model = model
        self.provider = provider
        self.system = system
        self.timeout = timeout
        self.hooks = hooks if hooks is not None else AgentHooks()
        self.max_messages = max_messages
        self.extra_body = extra_body
        self._cache = make_cache(cache_dir) if cache_dir is not None else None

        # Resolve provider config lazily on first run
        self._provider_config = resolve_provider(provider)

        # Cached conversation from the last run() call
        self._conversation: list[Message] = []

        # Snapshot the global tool registry at construction time so that
        # later @tool registrations don't affect this agent's tool set.
        self._global_tools: dict[str, ToolInfo] = {t.name: t for t in list_tools()}

        # Per-agent tools take name-priority over global ones.
        self._extra_tools: dict[str, ToolInfo] = {}
        if tools:
            for t in tools:
                self._extra_tools[t.name] = t

        # Interrupt support
        self._stop_event = asyncio.Event()
        self._current_task: asyncio.Task[None] | None = None
        self._aborted = False

    # ── public API ──────────────────────────────────────────────

    @property
    def conversation(self) -> list[Message]:
        """The conversation accumulated during the last ``run()`` call.

        Returns a shallow copy so callers can inspect and reuse messages
        without risk of mutating the agent's internal state mid-run.
        """
        return list(self._conversation)

    @property
    def messages(self) -> list[Message]:
        """Alias for ``conversation`` — available inside hook callbacks."""
        return list(self._conversation)

    def reset(self) -> None:
        """Clear conversation history and reset the stop flag."""
        self._conversation.clear()
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
          1. Sends conversation to the LLM (streaming).
          2. If the LLM returns tool calls, executes each (with timeout).
          3. Appends results and repeats.
          4. Stops when the LLM returns a non-tool-call response.

        Streaming notes:
          - Intermediate content delta messages have ``partial=True``.
          - The final assembled message for each LLM turn has ``partial=False``.
          - Tool result messages always have ``partial=False``.
          - ``max_messages`` counts only non-partial (complete) messages.

        After the loop, the full conversation is available at
        ``agent.conversation`` so you can inspect or restart::

            async for msg in agent.run([Message(role="user", content="Hi")]):
                ...

            # Restart with a better model, keeping history
            agent.model = "better-model"
            async for msg in agent.run(agent.conversation):
                ...

        Parameters
        ----------
        messages :
            Initial user/system messages for this run.
        usage :
            Optional mutable ``Usage`` object; cumulative token counts are
            added to it after each LLM turn (requires provider support).
        """
        self._stop_event.clear()
        self._aborted = False
        self._current_task = asyncio.current_task()
        self._conversation = list(messages)

        # Prepend system prompt only if not already present, so that passing
        # agent.conversation back into run() on restart does not duplicate it.
        if self.system and not (
            self._conversation and self._conversation[0].role == "system"
        ):
            self._conversation.insert(
                0, Message(role="system", content=self.system)
            )

        self._complete_yielded = 0
        self._max_messages_reached = False
        logger.debug("Agent.run starting: model=%s provider=%s", self.model, self.provider)
        await _safe_hook(self.hooks, "on_before_agent", self)
        try:
            while not self._stop_event.is_set():
                await _safe_hook(self.hooks, "on_before_turn", self)
                # ── 1. LLM call ──────────────────────────────────
                async for msg in self._stream_llm_response(usage):
                    yield msg

                if (
                    self._stop_event.is_set()
                    or self._max_messages_reached
                    or self._llm_last_chunk is None
                ):
                    return

                assistant_msg = self._llm_last_chunk
                self._conversation.append(assistant_msg)

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
                    *[
                        self._exec_single_tool(tc)
                        for tc in assistant_msg.tool_calls
                    ]
                )

                async for msg in self._emit_tool_results(results):
                    yield msg
                if self._max_messages_reached:
                    return

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

    def _check_max_messages(self) -> Message | None:
        """Increment complete counter and return a stop message if the limit is reached,
        else None."""
        self._complete_yielded += 1
        if self.max_messages > 0 and self._complete_yielded >= self.max_messages:
            return Message(
                role="assistant",
                content=(
                    f"[Agent stopped: reached max messages "
                    f"({self.max_messages})]"
                ),
                model=self.model,
            )
        return None

    async def _emit_tool_results(
        self, results: list[tuple[ToolCall, str, float]]
    ) -> AsyncIterator[Message]:
        """Yield tool result messages, appending each to the conversation.

        Stops early with a stop message if ``max_messages`` is reached.
        The caller should return from ``run()`` after this iterator
        completes — if the limit was hit the stop message is the last
        item yielded.
        """
        for tc, result_content, tool_duration in results:
            tool_msg = Message(
                role="tool",
                content=result_content,
                tool_call_id=tc.id,
                name=tc.function.name,
                duration=tool_duration,
            )
            self._conversation.append(tool_msg)
            yield tool_msg
            stop_msg = self._check_max_messages()
            if stop_msg is not None:
                yield stop_msg
                self._max_messages_reached = True
                return

    async def _stream_llm_response(
        self, usage: Usage | None
    ) -> AsyncIterator[Message]:
        """Run one LLM streaming turn, yielding partial and complete messages.

        Wraps the ``stream_chat`` call and ``CancelledError`` handling so the
        caller's loop stays flat.  After iteration, the final assembled
        ``Message`` is stored in ``self._llm_last_chunk`` (or ``None`` if the
        generator exited without receiving any chunk).
        """
        self._llm_last_chunk = None
        injected = await _safe_hook(self.hooks, "on_before_llm", self)
        if injected is not None:
            self._llm_last_chunk = injected
            yield injected
            replacement = await _safe_hook(self.hooks, "on_after_llm", self, injected)
            if replacement is not None:
                self._llm_last_chunk = replacement
            self._complete_yielded += 1
            return
        try:
            async for chunk in stream_chat(
                base_url=self._provider_config.base_url,
                api_key=self._provider_config.api_key,
                model=self.model,
                messages=_dump_messages(self._conversation),
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
                    self._complete_yielded += 1
                    if self.max_messages > 0 and self._complete_yielded >= self.max_messages:
                        yield Message(
                            role="assistant",
                            content=(
                                f"[Agent stopped: reached max messages "
                                f"({self.max_messages})]"
                            ),
                            model=self.model,
                        )
                        self._max_messages_reached = True
                        return
        except asyncio.CancelledError:
            if self._stop_event.is_set():
                return
            raise

    async def _exec_single_tool(
        self, tc: ToolCall
    ) -> tuple[ToolCall, str, float]:
        """Execute a single tool call and return (call, result, duration)."""
        name = tc.function.name
        try:
            args: dict[str, Any] = (
                json.loads(tc.function.arguments) if tc.function.arguments else {}
            )
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse arguments for tool '%s': %s", name, exc)
            return tc, f"Error: failed to parse arguments for '{name}': {exc}", 0.0

        injected = await _safe_hook(self.hooks, "on_before_tool", self, name, args)
        if injected is not None:
            logger.debug("Tool '%s' result injected by on_before_tool hook", name)
            await _safe_hook(self.hooks, "on_after_tool", self, name, args, injected)
            return tc, injected, 0.0

        info = self._resolve_tool(name)
        if info is None:
            logger.warning("Unknown tool requested: '%s'", name)
            result = (
                f"Error: unknown tool '{name}'. "
                f"Available tools: "
                f"{', '.join(t.name for t in self._all_tools())}"
            )
            await _safe_hook(self.hooks, "on_after_tool", self, name, args, result)
            return tc, result, 0.0

        logger.info("Tool call: %s(%s)", name, tc.function.arguments or "")
        t0 = time.perf_counter()
        result = await self._run_tool(info, args)
        duration = time.perf_counter() - t0
        logger.info("Tool '%s' completed in %.2fs", name, duration)

        await _safe_hook(self.hooks, "on_after_tool", self, name, args, result)
        return tc, result, duration

    # ── internals ──────────────────────────────────────────────

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        """Return OpenAI tool schemas for all known tools, or None."""
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

    def _all_tools(self) -> list[ToolInfo]:
        """Return all known tools (construction-time snapshot + extra)."""
        # extra_tools takes priority; global snapshot fills in the rest.
        seen: set[str] = set()
        result: list[ToolInfo] = []
        for t in self._extra_tools.values():
            seen.add(t.name)
            result.append(t)
        for t in self._global_tools.values():
            if t.name not in seen:
                seen.add(t.name)
                result.append(t)
        return result

    def _resolve_tool(self, name: str) -> ToolInfo | None:
        """Resolve a tool by name (extra takes priority, then construction-time snapshot)."""
        if name in self._extra_tools:
            return self._extra_tools[name]
        return self._global_tools.get(name)

    async def _run_tool(self, info: ToolInfo, args: dict[str, Any]) -> str:
        """Execute a tool with timeout and cancellation support."""
        schema = info.parameters
        allowed = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))
        missing = required - set(args.keys())
        unknown = set(args.keys()) - allowed
        if missing:
            return (
                f"Error: tool '{info.name}' missing required arguments: "
                f"{', '.join(sorted(missing))}"
            )
        if unknown:
            return (
                f"Error: tool '{info.name}' received unexpected arguments: "
                f"{', '.join(sorted(unknown))}"
            )

        try:
            async with asyncio.timeout(self.timeout):
                result = await info.fn(**args)
            return str(result)
        except asyncio.TimeoutError:
            logger.error("Tool '%s' timed out after %.1fs", info.name, self.timeout)
            return f"Error: tool '{info.name}' timed out after {self.timeout}s"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Tool '%s' raised %s: %s", info.name, type(exc).__name__, exc)
            return f"Error in tool '{info.name}': {exc}"


def _dump_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize messages to the dict format the API expects."""
    result: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            d["name"] = m.name
        result.append(d)
    return result
