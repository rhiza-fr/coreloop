"""The Agent — orchestrates the LLM loop with tool execution."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from ._client import stream_chat
from ._provider import resolve_provider
from ._tool import ToolInfo, list_tools
from ._types import Message, ToolCall, Usage


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
        max_turns: int = 20,
        max_messages: int = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        # Public — safe to read/write between runs
        self.model = model
        self.provider = provider
        self.system = system
        self.timeout = timeout
        self.max_turns = max_turns
        self.max_messages = max_messages
        self.extra_body = extra_body

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

    # ── public API ──────────────────────────────────────────────

    @property
    def conversation(self) -> list[Message]:
        """The conversation accumulated during the last ``run()`` call.

        Returns a shallow copy so callers can inspect and reuse messages
        without risk of mutating the agent's internal state mid-run.
        """
        return list(self._conversation)

    def reset(self) -> None:
        """Clear conversation history and reset the stop flag."""
        self._conversation.clear()
        self._stop_event.clear()

    def stop(self) -> None:
        """Signal the agent to stop as soon as possible.

        Sets the stop flag and cancels the current ``run()`` task (if any),
        which will interrupt any in-flight LLM call or tool execution.
        """
        self._stop_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    @property
    def stopped(self) -> bool:
        """Whether stop() has been called."""
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

        turns = 0
        complete_yielded = 0  # counts only non-partial messages
        try:
            while not self._stop_event.is_set():
                if turns >= self.max_turns:
                    yield Message(
                        role="assistant",
                        content=f"[Agent stopped: reached max turns ({self.max_turns})]",
                    )
                    return
                turns += 1

                # ── 1. LLM call ──────────────────────────────────
                assistant_msg: Message | None = None
                try:
                    last_chunk: Message | None = None
                    async for chunk in stream_chat(
                        base_url=self._provider_config.base_url,
                        api_key=self._provider_config.api_key,
                        model=self.model,
                        messages=_dump_messages(self._conversation),
                        tools=self._tool_schemas(),
                        timeout=self.timeout,
                        extra_body=self.extra_body,
                        usage=usage,
                    ):
                        if self._stop_event.is_set():
                            return
                        last_chunk = chunk
                        if chunk.partial:
                            yield chunk
                        else:
                            yield chunk.model_copy()
                            complete_yielded += 1
                            if self.max_messages > 0 and complete_yielded >= self.max_messages:
                                yield Message(
                                    role="assistant",
                                    content=(
                                        f"[Agent stopped: reached max messages "
                                        f"({self.max_messages})]"
                                    ),
                                )
                                return
                    assistant_msg = last_chunk
                except asyncio.CancelledError:
                    if self._stop_event.is_set():
                        return
                    raise

                if assistant_msg is None:
                    return

                self._conversation.append(assistant_msg)

                # ── 2. Check for tool calls ──────────────────────
                tool_calls = assistant_msg.tool_calls
                if not tool_calls:
                    return  # normal completion — no tools requested

                # ── 3. Execute tools concurrently ───────────────
                if self._stop_event.is_set():
                    return

                async def _exec(tc: ToolCall) -> tuple[ToolCall, str]:
                    info = self._resolve_tool(tc.function.name)
                    if info is None:
                        return tc, (
                            f"Error: unknown tool '{tc.function.name}'. "
                            f"Available tools: "
                            f"{', '.join(t.name for t in self._all_tools())}"
                        )
                    return tc, await self._run_tool(info, tc)

                results = await asyncio.gather(*[_exec(tc) for tc in tool_calls])

                for tc, result_content in results:
                    tool_msg = Message(
                        role="tool",
                        content=result_content,
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                    self._conversation.append(tool_msg)
                    yield tool_msg
                    complete_yielded += 1
                    if self.max_messages > 0 and complete_yielded >= self.max_messages:
                        yield Message(
                            role="assistant",
                            content=(
                                f"[Agent stopped: reached max messages "
                                f"({self.max_messages})]"
                            ),
                        )
                        return
        finally:
            self._current_task = None

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

    async def _run_tool(self, info: ToolInfo, tc: ToolCall) -> str:
        """Execute a tool with timeout and cancellation support."""
        try:
            args = (
                json.loads(tc.function.arguments)
                if tc.function.arguments
                else {}
            )
        except json.JSONDecodeError as exc:
            return f"Error: failed to parse arguments for '{info.name}': {exc}"

        # Validate args against the tool's JSON Schema before calling.
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
            return (
                f"Error: tool '{info.name}' timed out after {self.timeout}s"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
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
