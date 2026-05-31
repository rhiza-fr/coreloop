"""HTTP client for OpenAI-compatible chat completion endpoints with SSE streaming.

Note: Ollama via the openAI endpoint does not stream thinking models. They arrive at the end.

"""

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from ._cache import request_key
from .types import FunctionCall, Message, ToolCall, Usage

logger = logging.getLogger(__name__)


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
    llm_extra_body: dict[str, Any] | None = None,
    usage: Usage | None = None,
    cache: Any = None,
) -> AsyncIterator[Message]:
    """Stream chat completion chunks from an OpenAI-compatible API.

    Yields progressively richer *assistant* ``Message`` objects.
    The last yielded message for a turn will have either ``content`` set
    **or** ``tool_calls`` populated (or both).
    """
    cache_key: str | None = None
    if cache is not None:
        cache_key = request_key(model, messages, tools, llm_extra_body)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for model=%s messages=%d", model, len(messages))
            yield Message.model_validate_json(cached)
            return

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
    body["stream_options"] = {"include_usage": True}
    if llm_extra_body:
        for k, v in llm_extra_body.items():
            if k not in _PROTECTED:
                body[k] = v

    url = f"{base_url.rstrip('/')}{_CHAT_PATH}"

    accumulated_content: str | None = ""
    accumulated_reasoning: str | None = ""
    accumulated_tool_calls: list[ToolCall] | None = None
    # For incremental tool-call building: index -> partial ToolCall
    partials: dict[int, dict[str, Any]] = {}
    captured_usage: Usage | None = None
    # Assembled on finish_reason; held until [DONE] so usage chunk can arrive first
    pending_message: Message | None = None

    _owned = client is None
    _verify = base_url.startswith("https://")
    _session = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout), verify=_verify)
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
            async for line in resp.aiter_lines():
                # logger.debug("SSE %s", line)  # See raw response from the model
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

                # --- reasoning delta (thinking tokens, accumulated silently) ---
                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_delta:
                    accumulated_reasoning = (accumulated_reasoning or "") + reasoning_delta

                # --- content delta ---
                if delta.get("content"):
                    accumulated_content = (accumulated_content or "") + delta["content"]
                    yield Message(
                        role="assistant",
                        content=accumulated_content,
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

        # Yield the final message after [DONE] so usage is populated
        _duration = time.perf_counter() - _t0
        logger.debug(
            "Response from %s in %.2fs -- usage: %s",
            model,
            _duration,
            captured_usage,
        )
        final_message = (
            pending_message.model_copy(update={"usage": captured_usage, "duration": _duration})
            if pending_message is not None
            else Message(
                role="assistant",
                content=accumulated_content or None,
                tool_calls=accumulated_tool_calls,
                reasoning=accumulated_reasoning or None,
                usage=captured_usage,
                duration=_duration,
                model=model,
            )
        )
        if cache is not None and cache_key is not None:
            cache.set(cache_key, final_message.model_dump_json())
        yield final_message
    finally:
        if _owned:
            await _session.aclose()
