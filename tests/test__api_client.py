"""Tests for _tool_schemas and OpenAIBackend in _api_client."""

import httpx
import pytest

from minimal_agent._api_client import OpenAIBackend, _tool_schemas
from minimal_agent.tool_registry import ToolInfo


def _make_tool(name: str, description: str = "desc") -> ToolInfo:
    """Build a minimal ToolInfo for testing."""
    async def fn() -> str:
        return "ok"
    return ToolInfo(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
        fn=fn,
    )


# -- _tool_schemas -------------------------------------------------------------


def test_tool_schemas_none_input():
    """None input returns None."""
    assert _tool_schemas(None) is None


def test_tool_schemas_empty_list():
    """An empty list returns None."""
    assert _tool_schemas([]) is None


def test_tool_schemas_single_tool():
    """A single tool is wrapped in the OpenAI function-call schema."""
    tool = _make_tool("my_tool", "does something")
    result = _tool_schemas([tool])
    assert result is not None
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "my_tool"
    assert result[0]["function"]["description"] == "does something"


def test_tool_schemas_multiple_tools():
    """Multiple tools are all included in the output."""
    tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
    result = _tool_schemas(tools)
    assert result is not None
    assert len(result) == 3
    assert {r["function"]["name"] for r in result} == {"a", "b", "c"}


# -- OpenAIBackend -------------------------------------------------------------


def test_openai_backend_endpoint():
    """endpoint() appends the chat-completions path to the base URL."""
    backend = OpenAIBackend()
    assert backend.endpoint("http://localhost:11434/v1") == (
        "http://localhost:11434/v1/chat/completions"
    )


def test_openai_backend_endpoint_strips_trailing_slash():
    """Trailing slash on base_url is removed before appending the path."""
    backend = OpenAIBackend()
    assert backend.endpoint("http://localhost/v1/") == (
        "http://localhost/v1/chat/completions"
    )


def test_openai_backend_headers_no_api_key():
    """Without an API key, headers omit the Authorization field."""
    backend = OpenAIBackend()
    headers = backend.headers(None)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_openai_backend_headers_with_api_key():
    """With an API key, headers include a Bearer Authorization."""
    backend = OpenAIBackend()
    headers = backend.headers("sk-test")
    assert headers["Authorization"] == "Bearer sk-test"


def test_openai_backend_build_body_basic():
    """build_body produces a dict with model, messages, stream, and stream_options."""
    from minimal_agent.types import Message

    backend = OpenAIBackend()
    body = backend.build_body(
        model="gpt-4o",
        messages=[Message(role="user", content="hi")],
        tools=None,
        extra=None,
    )
    assert body["model"] == "gpt-4o"
    assert body["stream"] is True
    assert "messages" in body
    assert "tools" not in body


def test_openai_backend_build_body_with_tools():
    """build_body includes a tools list when tools are provided."""
    from minimal_agent.types import Message

    backend = OpenAIBackend()
    body = backend.build_body(
        model="gpt-4o",
        messages=[Message(role="user", content="hi")],
        tools=[_make_tool("read")],
        extra=None,
    )
    assert "tools" in body
    assert body["tools"][0]["function"]["name"] == "read"


def test_openai_backend_build_body_extra_keys():
    """Extra body keys are forwarded unless they are protected."""
    from minimal_agent.types import Message

    backend = OpenAIBackend()
    body = backend.build_body(
        model="m",
        messages=[Message(role="user", content="hi")],
        tools=None,
        extra={"temperature": 0.5, "model": "ignored"},
    )
    assert body["temperature"] == 0.5
    assert body["model"] == "m"  # protected key not overwritten


# -- parse_stream --------------------------------------------------------------


def _sse_response(*chunks: str) -> httpx.Response:
    """Build a fake httpx.Response that yields SSE lines."""
    body = "".join(chunks).encode()
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body,
    )


def _chunk(content: str, finish: bool = False) -> str:
    """Build a single SSE content-delta chunk."""
    import json
    return (
        "data: "
        + json.dumps({
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop" if finish else None}],
        })
        + "\n\n"
    )


def _tool_chunk(name: str, args: str, call_id: str = "t1") -> str:
    """Build a single SSE tool-call chunk."""
    import json
    return (
        "data: "
        + json.dumps({
            "id": "c2",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]}, "finish_reason": "tool_calls"}],
        })
        + "\n\n"
    )


def _done() -> str:
    """Return the SSE [DONE] sentinel."""
    return "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_parse_stream_text_content():
    """parse_stream assembles streamed text chunks into a final message."""
    resp = _sse_response(_chunk("Hello"), _chunk(" world", finish=True), _done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    final = messages[-1]
    assert final.content == "Hello world"
    assert final.tool_calls is None


@pytest.mark.asyncio
async def test_parse_stream_tool_call():
    """parse_stream assembles a tool-call chunk into a ToolCall message."""
    resp = _sse_response(_tool_chunk("read", '{"path":"x"}'), _done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    final = messages[-1]
    assert final.tool_calls is not None
    assert final.tool_calls[0].function.name == "read"


@pytest.mark.asyncio
async def test_parse_stream_empty_response():
    """A [DONE]-only response yields a single empty final message."""
    resp = _sse_response(_done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    assert len(messages) == 1
    assert messages[0].content is None
    assert messages[0].tool_calls is None


@pytest.mark.asyncio
async def test_parse_stream_ignores_invalid_json():
    """Lines with invalid JSON are silently skipped."""
    import json
    bad_line = "data: {not valid json}\n\n"
    good_line = "data: " + json.dumps({
        "id": "c1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
    }) + "\n\n"
    resp = _sse_response(bad_line, good_line, _done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    final = messages[-1]
    assert "ok" in (final.content or "")


@pytest.mark.asyncio
async def test_parse_stream_usage_chunk_captured():
    """A usage chunk (no choices) is captured and attached to the final message."""
    import json

    usage_chunk = "data: " + json.dumps({
        "id": "c1", "object": "chat.completion.chunk",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }) + "\n\n"
    resp = _sse_response(_chunk("hi", finish=True), usage_chunk, _done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    final = messages[-1]
    assert final.usage is not None
    assert final.usage.prompt_tokens == 10


@pytest.mark.asyncio
async def test_parse_stream_reasoning_delta():
    """Reasoning/thinking content is accumulated in the reasoning field."""
    import json

    reasoning_chunk = "data: " + json.dumps({
        "id": "c1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"reasoning_content": "think..."}, "finish_reason": None}],
    }) + "\n\n"
    resp = _sse_response(reasoning_chunk, _chunk("answer", finish=True), _done())
    backend = OpenAIBackend()
    messages = []
    async for msg in backend.parse_stream(resp, model="m", usage=None):
        messages.append(msg)
    final = messages[-1]
    assert final.reasoning == "think..."
    assert final.content == "answer"
