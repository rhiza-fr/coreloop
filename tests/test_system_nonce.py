"""Tests for the per-run system-prompt nonce (system_nonce).

The nonce prepends a unique token to the system message at the start of each
run so independent requests sharing a system prompt no longer share a prefix --
defeating Ollama's cross-request prompt-cache reuse.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

import coreloop.agent as agent_mod
from coreloop import Agent, Message
from coreloop.types import Usage


class _Capture:
    """Records the messages handed to stream_chat on each LLM call."""

    def __init__(self) -> None:
        self.seen: list[list[Message]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_stream_chat(
            *, messages: list[Message], **_: Any
        ) -> AsyncIterator[Message]:
            self.seen.append([m.model_copy() for m in messages])
            yield Message(role="assistant", content="ok", usage=Usage())

        monkeypatch.setattr(agent_mod, "stream_chat", fake_stream_chat)


@pytest.mark.asyncio
async def test_nonce_prepended_to_system_message(monkeypatch: pytest.MonkeyPatch):
    """With system_nonce, the system message sent to the LLM is nonce-prefixed."""
    cap = _Capture()
    cap.install(monkeypatch)
    agent = Agent(model="m", system="You are helpful.", system_nonce=True, cache_dir=None)

    async for _ in agent.run([Message(role="user", content="hi")]):
        pass

    sent_system = cap.seen[0][0]
    assert sent_system.role == "system"
    assert sent_system.content is not None
    assert sent_system.content.startswith("[run:")
    assert sent_system.content.endswith("You are helpful.")


@pytest.mark.asyncio
async def test_nonce_differs_between_runs(monkeypatch: pytest.MonkeyPatch):
    """Each run gets a distinct nonce, so two runs share no system prefix."""
    cap = _Capture()
    cap.install(monkeypatch)
    agent = Agent(model="m", system="You are helpful.", system_nonce=True, cache_dir=None)

    async for _ in agent.run([Message(role="user", content="hi")]):
        pass
    async for _ in agent.run([Message(role="user", content="hi")]):
        pass

    first = cap.seen[0][0].content
    second = cap.seen[-1][0].content
    assert first is not None and second is not None
    assert first != second
    assert first.split("]")[0] != second.split("]")[0]


@pytest.mark.asyncio
async def test_no_nonce_when_disabled(monkeypatch: pytest.MonkeyPatch):
    """Without system_nonce the system message is sent verbatim."""
    cap = _Capture()
    cap.install(monkeypatch)
    agent = Agent(model="m", system="You are helpful.", cache_dir=None)

    async for _ in agent.run([Message(role="user", content="hi")]):
        pass

    assert cap.seen[0][0].content == "You are helpful."
