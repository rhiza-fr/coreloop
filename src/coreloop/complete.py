"""One-shot, schema-constrained completion -- no tools, no loop."""

from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel

from ._api_client import OPENAI_BACKEND, Backend, stream_chat
from ._cache import make_cache
from .agent import _DEFAULT_CACHE_DIR
from .types import Message, Usage

__all__ = ["complete"]


@overload
async def complete[T: BaseModel](
    prompt: str | list[Message],
    *,
    model: str,
    schema: type[T],
    base_url: str = ...,
    api_key: str | None = ...,
    system: str | None = ...,
    timeout: float = ...,
    llm_extra_body: dict[str, Any] | None = ...,
    usage: Usage | None = ...,
    cache_dir: Path | str | None = ...,
    backend: Backend = ...,
) -> T: ...


@overload
async def complete[T: BaseModel](
    prompt: str | list[Message],
    *,
    model: str,
    schema: None = None,
    base_url: str = ...,
    api_key: str | None = ...,
    system: str | None = ...,
    timeout: float = ...,
    llm_extra_body: dict[str, Any] | None = ...,
    usage: Usage | None = ...,
    cache_dir: Path | str | None = ...,
    backend: Backend = ...,
) -> Message: ...


async def complete[T: BaseModel](
    prompt: str | list[Message],
    *,
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str | None = None,
    system: str | None = None,
    schema: type[T] | None = None,
    timeout: float = 300.0,
    llm_extra_body: dict[str, Any] | None = None,
    usage: Usage | None = None,
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
    backend: Backend = OPENAI_BACKEND,
) -> Message | T:
    """One LLM call, no tools, no loop. With *schema*, returns a validated instance.

    *prompt* is either a string (wrapped into a single user message) or a list of
    ``Message`` objects sent as-is -- the latter is the "extract from
    ``agent.messages``" path. *system*, when given, is prepended as a system message.

    With *schema*, output is constrained via the backend's ``response_format``
    and the final content is validated with ``schema.model_validate_json``.
    ``ValidationError`` and ``json.JSONDecodeError`` propagate to the caller.

    Note: a provider may silently ignore request fields it does not know
    (Ollama's ``/v1`` endpoint drops unknown keys with HTTP 200), so anything
    passed through *llm_extra_body* should be verified against observed
    behaviour rather than assumed to take effect.
    """
    messages = [Message(role="user", content=prompt)] if isinstance(prompt, str) else list(prompt)
    if system is not None:
        messages = [Message(role="system", content=system), *messages]

    cache = make_cache(cache_dir) if cache_dir is not None else None

    final: Message | None = None
    async for msg in stream_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        tools=None,
        schema=schema,
        timeout=timeout,
        llm_extra_body=llm_extra_body,
        usage=usage,
        cache=cache,
        backend=backend,
    ):
        if not msg.partial:
            final = msg

    if final is None:
        raise ValueError(f"No response from {model}")
    if schema is None:
        return final
    if not final.content:
        raise ValueError(f"Empty response content from {model} while a schema was set")
    return schema.model_validate_json(final.content)
