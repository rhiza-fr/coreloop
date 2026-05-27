"""HTTP client for OpenAI-compatible chat completion endpoints with SSE streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ._types import Message, ToolCall, FunctionCall, Usage


_CHAT_PATH = "/chat/completions"


async def stream_chat(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 60.0,
    client: httpx.AsyncClient | None = None,
    extra_body: dict[str, Any] | None = None,
    usage: Usage | None = None,
) -> AsyncIterator[Message]:
    """Stream chat completion chunks from an OpenAI-compatible API.

    Yields progressively richer *assistant* ``Message`` objects.
    The last yielded message for a turn will have either ``content`` set
    **or** ``tool_calls`` populated (or both).
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _PROTECTED = frozenset({"model", "messages", "stream", "tools", "stream_options"})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    if usage is not None:
        body["stream_options"] = {"include_usage": True}
    if extra_body:
        for k, v in extra_body.items():
            if k not in _PROTECTED:
                body[k] = v

    url = f"{base_url.rstrip('/')}{_CHAT_PATH}"

    accumulated_content: str | None = ""
    accumulated_reasoning: str | None = ""
    accumulated_tool_calls: list[ToolCall] | None = None
    # For incremental tool-call building: index → partial ToolCall
    partials: dict[int, dict[str, Any]] = {}

    _owned = client is None
    _session = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    try:
        async with _session.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
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
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # --- reasoning delta (thinking tokens, accumulated silently) ---
                if "reasoning" in delta and delta["reasoning"] is not None:
                    accumulated_reasoning = (accumulated_reasoning or "") + delta["reasoning"]

                # --- content delta ---
                if "content" in delta and delta["content"] is not None:
                    accumulated_content = (accumulated_content or "") + delta["content"]
                    yield Message(
                        role="assistant",
                        content=accumulated_content,
                        partial=True,
                    )

                # --- usage (present in stream_options usage chunk, no choices) ---
                if usage is not None and "usage" in chunk and chunk["usage"]:
                    u = chunk["usage"]
                    usage.prompt_tokens += u.get("prompt_tokens", 0)
                    usage.completion_tokens += u.get("completion_tokens", 0)
                    usage.total_tokens += u.get("total_tokens", 0)

                # --- tool call deltas ---
                raw_tool_calls = delta.get("tool_calls")
                if raw_tool_calls:
                    if accumulated_tool_calls is None:
                        accumulated_tool_calls = []
                    for tc_delta in raw_tool_calls:
                        idx = tc_delta.get("index", 0)
                        partial = partials.setdefault(idx, {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if "id" in tc_delta and tc_delta["id"]:
                            partial["id"] = tc_delta["id"]
                        if "function" in tc_delta:
                            fn = tc_delta["function"]
                            if "name" in fn and fn["name"]:
                                partial["function"]["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"]:
                                partial["function"]["arguments"] += fn["arguments"]

                # --- finalise on finish ---
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
                            for idx, p in sorted(partials.items())
                        ]
                        partials.clear()

                    yield Message(
                        role="assistant",
                        content=accumulated_content or None,
                        tool_calls=accumulated_tool_calls,
                        reasoning=accumulated_reasoning or None,
                    )
                    return

        # Fallback yield if stream ends without finish_reason
        yield Message(
            role="assistant",
            content=accumulated_content or None,
            tool_calls=accumulated_tool_calls,
            reasoning=accumulated_reasoning or None,
        )
    finally:
        if _owned:
            await _session.aclose()
