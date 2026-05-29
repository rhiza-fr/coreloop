"""Message and tool-call type models matching OpenAI's chat format."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """A function call inside a tool call."""

    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    """A tool call emitted by the assistant."""

    id: str
    type: str = "function"
    function: FunctionCall


class Message(BaseModel):
    """A single message in the conversation, OpenAI-compatible shape.

    Usage:
        Message(role="user", content="Hello")
        Message(role="assistant", content=None, tool_calls=[...])
        Message(role="tool", content="result", tool_call_id="...", name="read")

    ``reasoning`` is a streaming-only field used by thinking models (Qwen3,
    DeepSeek).  It is **not** included in the conversation history sent
    back to the API (``_dump_messages`` omits it).

    ``partial`` is True for incremental content delta messages emitted while
    the LLM is still generating.  The final assembled message for a turn has
    ``partial=False``.  Consumers that only want complete messages should skip
    messages where ``partial is True``.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    partial: bool = False
    usage: "Usage | None" = None
    duration: float | None = None
    model: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Usage(BaseModel):
    """Token usage reported by the model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
