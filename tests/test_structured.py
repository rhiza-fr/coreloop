"""Tests for to_strict_schema, response_format, and complete()."""

import importlib
import json
from enum import Enum

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError

from coreloop import Message, complete, to_strict_schema
from coreloop._api_client import OpenAIBackend, stream_chat
from coreloop._cache import request_key

# ``coreloop.complete`` the module, not the re-exported function of the same name.
complete_mod = importlib.import_module("coreloop.complete")


class Flat(BaseModel):
    name: str
    count: int = Field(default=3, description="how many")


class Inner(BaseModel):
    kind: str


class Priority(str, Enum):
    low = "low"
    high = "high"


class Nested(BaseModel):
    inner: Inner
    items: list[Inner]
    tags: list[str]
    priority: Priority
    note: str | None = None


# -- to_strict_schema ----------------------------------------------------------


def test_strict_schema_flat_model():
    """A flat model gets additionalProperties=false and every property required."""
    s = to_strict_schema(Flat)
    assert s["additionalProperties"] is False
    assert sorted(s["required"]) == ["count", "name"]


def test_strict_schema_strips_title_and_default():
    """title and default are stripped at every level; description survives."""
    s = to_strict_schema(Flat)
    assert "title" not in s
    assert "title" not in s["properties"]["count"]
    assert "default" not in s["properties"]["count"]
    assert s["properties"]["count"]["description"] == "how many"


def test_strict_schema_nested_via_defs():
    """Nested models under $defs are tightened in place, and $ref is preserved."""
    s = to_strict_schema(Nested)
    assert s["properties"]["inner"] == {"$ref": "#/$defs/Inner"}
    inner = s["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["kind"]


def test_strict_schema_list_of_models():
    """list[T] keeps its array shape with the item $ref intact."""
    s = to_strict_schema(Nested)
    items = s["properties"]["items"]
    assert items["type"] == "array"
    assert items["items"] == {"$ref": "#/$defs/Inner"}


def test_strict_schema_optional_keeps_null_branch_and_stays_required():
    """An ``X | None`` field keeps its null branch and remains formally required."""
    s = to_strict_schema(Nested)
    assert {"type": "null"} in s["properties"]["note"]["anyOf"]
    assert "note" in s["required"]


def test_strict_schema_preserves_enums():
    """Enum values survive the rewrite."""
    s = to_strict_schema(Nested)
    assert s["$defs"]["Priority"]["enum"] == ["low", "high"]


# -- Backend.response_format ---------------------------------------------------


def test_response_format_fragment_is_well_formed():
    """The body fragment names the model, sets strict, and carries the schema."""
    frag = OpenAIBackend().response_format(Flat)
    rf = frag["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Flat"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == to_strict_schema(Flat)


# -- stream_chat merging -------------------------------------------------------


def _text_stream(content: str) -> bytes:
    """SSE bytes for a single-chunk text completion."""
    chunk = json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
        }
    )
    return f"data: {chunk}\n\ndata: [DONE]\n\n".encode()


def _capturing_client(content: str, sent: list[dict]) -> httpx.AsyncClient:
    """An httpx client that records request bodies and replies with *content*."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_text_stream(content),
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _drain(**kwargs) -> Message:
    """Run stream_chat and return its final (non-partial) message."""
    final = None
    async for msg in stream_chat(**kwargs):
        if not msg.partial:
            final = msg
    assert final is not None
    return final


@pytest.mark.asyncio
async def test_stream_chat_merges_schema_into_body():
    """With a schema, stream_chat merges response_format into the request body."""
    sent: list[dict] = []
    async with _capturing_client("{}", sent) as client:
        await _drain(
            base_url="http://x/v1",
            api_key=None,
            model="m",
            messages=[Message(role="user", content="hi")],
            schema=Flat,
            client=client,
        )
    assert sent[0]["response_format"]["json_schema"]["name"] == "Flat"


@pytest.mark.asyncio
async def test_stream_chat_omits_schema_when_none():
    """Without a schema, no response_format is sent."""
    sent: list[dict] = []
    async with _capturing_client("hi", sent) as client:
        await _drain(
            base_url="http://x/v1",
            api_key=None,
            model="m",
            messages=[Message(role="user", content="hi")],
            client=client,
        )
    assert "response_format" not in sent[0]


def test_cache_key_differs_by_schema():
    """Same messages with different schemas produce different cache keys."""
    backend = OpenAIBackend()
    messages = [Message(role="user", content="hi")]
    base = backend.build_body("m", messages, None, None)

    plain = request_key("http://x/v1", dict(base))
    with_flat = request_key("http://x/v1", {**base, **backend.response_format(Flat)})
    with_nested = request_key("http://x/v1", {**base, **backend.response_format(Nested)})

    assert len({plain, with_flat, with_nested}) == 3


# -- complete() ----------------------------------------------------------------


class _Recorder(list):
    """Recorded stream_chat calls, plus the canned reply the fake yields."""

    reply: dict


@pytest.fixture
def calls(monkeypatch):
    """Replace stream_chat inside complete() with a recorder; returns the call list."""
    recorded = _Recorder()
    recorded.reply = {"value": "hello"}

    async def fake_stream_chat(**kwargs):
        recorded.append(kwargs)
        yield Message(role="assistant", content=recorded.reply["value"], partial=True)
        yield Message(role="assistant", content=recorded.reply["value"])

    monkeypatch.setattr(complete_mod, "stream_chat", fake_stream_chat)
    return recorded


@pytest.mark.asyncio
async def test_complete_str_prompt_returns_message(calls):
    """A str prompt becomes one user message; without a schema a Message is returned."""
    result = await complete("hi", model="m", cache_dir=None)
    assert isinstance(result, Message)
    assert result.content == "hello"
    assert [(m.role, m.content) for m in calls[0]["messages"]] == [("user", "hi")]
    assert calls[0]["tools"] is None
    assert calls[0]["schema"] is None


@pytest.mark.asyncio
async def test_complete_system_is_prepended(calls):
    """`system` is prepended as a system message."""
    await complete("hi", model="m", system="be terse", cache_dir=None)
    sent = calls[0]["messages"]
    assert sent[0].role == "system" and sent[0].content == "be terse"
    assert sent[1].content == "hi"


@pytest.mark.asyncio
async def test_complete_message_list_sent_as_is(calls):
    """A list[Message] prompt is sent unchanged."""
    history = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    await complete(history, model="m", cache_dir=None)
    assert [m.content for m in calls[0]["messages"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_complete_with_schema_returns_validated_instance(calls):
    """With a schema, the validated model instance is returned."""
    calls.reply["value"] = '{"name":"x","count":7}'
    result = await complete("hi", model="m", schema=Flat, cache_dir=None)
    assert isinstance(result, Flat)
    assert result.name == "x"
    assert result.count == 7
    assert calls[0]["schema"] is Flat


@pytest.mark.asyncio
async def test_complete_propagates_validation_error(calls):
    """Non-conforming content raises ValidationError."""
    calls.reply["value"] = '{"count":7}'
    with pytest.raises(ValidationError):
        await complete("hi", model="m", schema=Flat, cache_dir=None)


@pytest.mark.asyncio
async def test_complete_empty_content_raises(calls):
    """Empty content while a schema is set raises."""
    calls.reply["value"] = ""
    with pytest.raises(ValueError, match="Empty response"):
        await complete("hi", model="m", schema=Flat, cache_dir=None)
