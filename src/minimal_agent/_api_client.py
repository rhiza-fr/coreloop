"""HTTP streaming client and the pluggable :class:`Backend` seam.

A *backend* owns everything provider-specific: the endpoint path, auth headers,
request-body shape (message + tool serialization), and the wire-stream parser
that turns raw response lines into canonical ``Message`` objects.  ``stream_chat``
is the shared shell around any backend -- it owns the httpx connection, response
caching, timing, and HTTP error handling.

Only one backend ships today (:class:`OpenAIBackend`, OpenAI-compatible chat
completions).  Adding another provider means implementing the four ``Backend``
methods; nothing else in the agent loop needs to change.

Note: Ollama via the OpenAI endpoint does not stream thinking models. They
arrive at the end.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Protocol

import httpx

from ._cache import request_key
from .tool_registry import ToolInfo
from .types import FunctionCall, Message, ToolCall, Usage, _dump_messages

logger = logging.getLogger(__name__)


class Backend(Protocol):
    """Provider-specific request shaping and stream parsing.

    Implement these four methods to add a provider. The agent loop and
    ``stream_chat`` shell are backend-agnostic.
    """

    def endpoint(self, base_url: str) -> str:
        """Return the full URL to POST to."""
        ...

    def headers(self, api_key: str | None) -> dict[str, str]:
        """Return request headers (content type, accept, auth)."""
        ...

    def build_body(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolInfo] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Serialize canonical messages + tools into the provider's request body."""
        ...

    def parse_stream(
        self, resp: httpx.Response, model: str, usage: Usage | None
    ) -> AsyncIterator[Message]:
        """Parse the streaming response into progressively richer ``Message`` objects.

        Yields ``partial=True`` messages as content/reasoning stream in, and
        exactly one final ``partial=False`` message (with ``usage`` populated)
        at the end. ``duration`` is left unset -- the shell attaches wall-clock
        timing. If *usage* is given, token counts are accumulated into it.
        """
        ...


def _tool_schemas(tools: list[ToolInfo] | None) -> list[dict[str, Any]] | None:
    """Wrap tools in the OpenAI ``{type: function, function: {...}}`` schema."""
    if not tools:
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
        for t in tools
    ]


class OpenAIBackend:
    """OpenAI-compatible chat-completions backend (the default)."""

    _CHAT_PATH = "/chat/completions"
    _PROTECTED = frozenset({"model", "messages", "stream", "tools", "stream_options"})

    def endpoint(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}{self._CHAT_PATH}"

    def headers(self, api_key: str | None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_body(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolInfo] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": _dump_messages(messages),
            "stream": True,
        }
        schemas = _tool_schemas(tools)
        if schemas:
            body["tools"] = schemas
        body["stream_options"] = {"include_usage": True}
        if extra:
            for k, v in extra.items():
                if k not in self._PROTECTED:
                    body[k] = v
        return body

    async def parse_stream(
        self, resp: httpx.Response, model: str, usage: Usage | None
    ) -> AsyncIterator[Message]:
        accumulated_content: str | None = ""
        accumulated_reasoning: str | None = ""
        accumulated_tool_calls: list[ToolCall] | None = None
        # For incremental tool-call building: index -> partial ToolCall
        partials: dict[int, dict[str, Any]] = {}
        captured_usage: Usage | None = None
        # Assembled on finish_reason; held until [DONE] so usage chunk can arrive first
        pending_message: Message | None = None

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # --- usage (arrives in a separate no-choices chunk after finish_reason) ---
            if "usage" in chunk and chunk["usage"]:
                u = chunk["usage"]
                pt = u.get("prompt_tokens", 0)
                ct = u.get("completion_tokens", 0)
                tt = u.get("total_tokens", 0)
                captured_usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
                if usage is not None:
                    usage.prompt_tokens += pt
                    usage.completion_tokens += ct
                    usage.total_tokens += tt

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            # --- reasoning delta (thinking tokens, streamed as partials) ---
            reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_delta:
                accumulated_reasoning = (accumulated_reasoning or "") + reasoning_delta
                yield Message(
                    role="assistant",
                    content=accumulated_content or None,
                    reasoning=accumulated_reasoning,
                    partial=True,
                    model=model,
                )

            # --- content delta ---
            if delta.get("content"):
                accumulated_content = (accumulated_content or "") + delta["content"]
                yield Message(
                    role="assistant",
                    content=accumulated_content,
                    reasoning=accumulated_reasoning or None,
                    partial=True,
                    model=model,
                )

            # --- tool call deltas ---
            raw_tool_calls = delta.get("tool_calls")
            if raw_tool_calls:
                if accumulated_tool_calls is None:
                    accumulated_tool_calls = []
                for tc_delta in raw_tool_calls:
                    idx = tc_delta.get("index", 0)
                    partial = partials.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if "id" in tc_delta and tc_delta["id"]:
                        partial["id"] = tc_delta["id"]
                    if "function" in tc_delta:
                        fn = tc_delta["function"]
                        if "name" in fn and fn["name"]:
                            partial["function"]["name"] += fn["name"]
                        if "arguments" in fn and fn["arguments"]:
                            partial["function"]["arguments"] += fn["arguments"]

            # --- assemble on finish, but keep reading for the usage chunk ---
            if finish_reason:
                if partials:
                    accumulated_tool_calls = [
                        ToolCall(
                            id=p["id"],
                            type=p["type"],
                            function=FunctionCall(
                                name=p["function"]["name"],
                                arguments=p["function"]["arguments"],
                            ),
                        )
                        for _idx, p in sorted(partials.items())
                    ]
                    partials.clear()
                pending_message = Message(
                    role="assistant",
                    content=accumulated_content or None,
                    tool_calls=accumulated_tool_calls,
                    reasoning=accumulated_reasoning or None,
                    model=model,
                )

        if pending_message is not None:
            yield pending_message.model_copy(update={"usage": captured_usage})
        else:
            yield Message(
                role="assistant",
                content=accumulated_content or None,
                tool_calls=accumulated_tool_calls,
                reasoning=accumulated_reasoning or None,
                usage=captured_usage,
                model=model,
            )


OPENAI_BACKEND: Backend = OpenAIBackend()
"""The default backend: OpenAI-compatible chat completions."""


async def stream_chat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[Message],
    tools: list[ToolInfo] | None = None,
    timeout: float = 60.0,
    client: httpx.AsyncClient | None = None,
    llm_extra_body: dict[str, Any] | None = None,
    usage: Usage | None = None,
    cache: Any = None,
    backend: Backend = OPENAI_BACKEND,
) -> AsyncIterator[Message]:
    """Stream chat completion chunks from *backend*, yielding ``Message`` objects.

    The shared shell around any :class:`Backend`: builds the request body,
    serves/stores cached responses, manages the httpx connection, attaches
    wall-clock ``duration`` to the final message, and raises on HTTP errors.

    Yields progressively richer *assistant* ``Message`` objects. The last yielded
    message for a turn has either ``content`` set **or** ``tool_calls`` populated
    (or both), plus ``usage`` and ``duration``.
    """
    body = backend.build_body(model, messages, tools, llm_extra_body)

    cache_key: str | None = None
    if cache is not None:
        cache_key = request_key(base_url, body)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for model=%s messages=%d", model, len(messages))
            yield Message.model_validate_json(cached)
            return

    url = backend.endpoint(base_url)
    headers = backend.headers(api_key)

    _owned = client is None
    _session = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    _t0 = time.perf_counter()
    logger.info("POST %s model=%s messages=%d", url, model, len(messages))
    try:
        async with _session.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                detail = resp.text[:500]
                msg = f"HTTP {resp.status_code} from {url}"
                if detail:
                    msg += f": {detail}"
                raise httpx.HTTPStatusError(msg, request=resp.request, response=resp)

            final: Message | None = None
            async for chunk in backend.parse_stream(resp, model, usage):
                if chunk.partial:
                    yield chunk
                else:
                    final = chunk

        if final is None:
            return
        _duration = time.perf_counter() - _t0
        final = final.model_copy(update={"duration": _duration})
        logger.debug("Response from %s in %.2fs -- usage: %s", model, _duration, final.usage)
        if cache is not None and cache_key is not None:
            cache.set(cache_key, final.model_dump_json())
        yield final
    finally:
        if _owned:
            await _session.aclose()
