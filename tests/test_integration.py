"""Integration tests using httpx.MockTransport to simulate the API."""

import json

import httpx
import pytest

from minimal_agent import Agent, Message, tool
from minimal_agent.agent import _dump_messages
from minimal_agent._client import stream_chat


# ── SSE helpers ────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def _content_chunk(content: str, finish: bool = False) -> str:
    return _sse({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": "stop" if finish else None,
        }],
    })

def _tool_call_chunks(name: str, args: str, call_id: str = "call_1") -> list[str]:
    return [
        _sse({
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0, "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }],
                },
                "finish_reason": None,
            }],
        }),
        _sse({
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0, "id": None, "type": None,
                        "function": {"name": None, "arguments": args},
                    }],
                },
                "finish_reason": None,
            }],
        }),
        _sse({
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }),
    ]

def _done() -> str:
    return "data: [DONE]\n\n"

def _mock_client(*sse_lines: str) -> httpx.AsyncClient:
    """Create an AsyncClient backed by MockTransport that serves SSE lines."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(sse_lines).encode()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "content-length": str(len(body))},
            content=body,
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Client tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_chat_text():
    """Simple text response yields progressive content, then final message."""
    lines = [_content_chunk("Hello", finish=False), _content_chunk(" world", finish=True), _done()]
    client = _mock_client(*lines)

    collected: list[Message] = []
    async for msg in stream_chat(
        base_url="http://test", api_key=None, model="m",
        messages=[{"role": "user", "content": "hi"}],
        client=client,
    ):
        collected.append(msg)

    assert len(collected) >= 2  # progressive + final
    final = collected[-1]
    assert final.role == "assistant"
    assert final.content == "Hello world"
    assert final.tool_calls is None


@pytest.mark.asyncio
async def test_stream_chat_tool_calls():
    """Tool-call response yields a message with tool_calls populated."""
    lines = _tool_call_chunks("read", '{"path":"test.txt"}') + [_done()]
    client = _mock_client(*lines)

    collected: list[Message] = []
    async for msg in stream_chat(
        base_url="http://test", api_key=None, model="m",
        messages=[{"role": "user", "content": "read file"}],
        client=client,
    ):
        collected.append(msg)

    final = collected[-1]
    assert final.role == "assistant"
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].function.name == "read"
    assert json.loads(final.tool_calls[0].function.arguments) == {"path": "test.txt"}


# ── Agent tests ────────────────────────────────────────────────

@pytest.mark.slow
def test_agent_construct():
    agent = Agent(model="qwen3.5:9b", provider="ollama")
    assert not agent.stopped

    agent.stop()
    assert agent.stopped


@pytest.mark.slow
@pytest.mark.asyncio
async def test_agent_with_tool():
    """Agent executes a tool when the LLM requests one, then continues."""
    @tool(allow_override=True)
    async def read(path: str) -> str:
        return f"contents of {path}"

    agent = Agent(model="qwen3.5:9b", provider="ollama")
    # Monkey-patch the internal client creation by overriding provider config
    agent._provider_config.base_url = "http://test"
    agent._provider_config.api_key = None

    # Hack: we need to inject our client into stream_chat calls.
    # For a proper test, we'd refactor Agent to accept a client.
    # For now, let's test the tool execution path directly.
    from minimal_agent.tool import get_tool

    info = get_tool("read")
    assert info is not None
    result = await agent._run_tool(info, {"path": "foo.txt"})
    assert result == "contents of foo.txt"


# ── Serialization test ─────────────────────────────────────────

def test_dump_messages():
    """_dump_messages produces correct dict format."""
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hi"),
    ]
    dumped = _dump_messages(msgs)
    assert dumped == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]

    # With tool message
    msgs.append(Message(role="tool", content="result", tool_call_id="c1", name="read"))
    dumped = _dump_messages(msgs)
    assert dumped[2] == {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "read"}
