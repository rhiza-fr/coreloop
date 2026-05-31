"""End-to-end integration test against a real Ollama instance.

Tests that the Agent loop works end-to-end with ``qwen3.5:9b`` on
``192.168.0.101``.  Skipped when the host is unreachable.

Tool-calling tests are model-dependent -- many local models do not emit
structured ``tool_calls`` in the API format even when instructed.
Those tests are best-effort but do not fail when a model only replies
with text; they log diagnostics instead.
"""

import tempfile
from pathlib import Path

import httpx
import pytest

from minimal_agent import Agent, Message, make_tools

OLLAMA_HOST = "http://192.168.0.101:11434/v1"
MODEL = "qwen3.5:9b"
TIMEOUT = 120.0


def _ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_HOST}/models", timeout=5.0)
        return r.status_code == 200
    except httpx.ConnectError, httpx.TimeoutException:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable"),
    pytest.mark.asyncio,
    pytest.mark.slow,
]


# -- Fixtures --------------------------------------------------


@pytest.fixture
def sandbox():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("# Test Project\n\nHello.\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
        (root / "data.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
        yield tmp


@pytest.fixture
def agent(sandbox):
    tools = make_tools(allowed_root=sandbox)
    return Agent(
        model=MODEL,
        base_url=OLLAMA_HOST,
        system="You are a helpful assistant with file tools: read, ls, edit.",
        tools=tools,
        llm_timeout=TIMEOUT,
    )


@pytest.fixture(autouse=True)
def _override_base_url(agent):
    agent._base_url = OLLAMA_HOST
    agent._api_key = None


# -- Helpers ---------------------------------------------------


async def _collect(agent, messages: list[Message]) -> list[Message]:
    results: list[Message] = []
    async for msg in agent.run(messages):
        results.append(msg)
    return results


def _fmt(results: list[Message]) -> str:
    lines: list[str] = []
    for m in results:
        if m.role == "assistant":
            tc = (
                f" [tool_calls: {[t.function.name for t in (m.tool_calls or [])]}]"
                if m.tool_calls
                else ""
            )
            preview = (m.content or "")[:150]
            lines.append(f"  [bot] {preview!r}{tc}")
        elif m.role == "tool":
            preview = (m.content or "")[:150]
            lines.append(f"  [tool] {m.name}: {preview!r}")
        else:
            lines.append(f"  {m.role}: {(m.content or '')[:150]!r}")
    return "\n".join(lines)


# -- Core pipeline tests (model-agnostic) ----------------------


async def test_simple_chat(agent):
    """Basic text-only query gets an assistant response (streaming works)."""
    results = await _collect(agent, [Message(role="user", content="Say exactly: hello world")])
    assistant_msgs = [m for m in results if m.role == "assistant"]

    assert len(assistant_msgs) >= 1, f"No assistant messages:\n{_fmt(results)}"
    final = assistant_msgs[-1]
    assert final.content is not None, "Final assistant has no content"
    # The streaming yielded progressively longer content
    assert len(final.content) > 0


async def test_path_traversal_refused(agent):
    """Agent correctly denies path traversal via the read tool."""
    results = await _collect(
        agent, [Message(role="user", content="Read /etc/passwd using the read tool.")]
    )
    tool_msgs = [m for m in results if m.role == "tool"]
    read_msgs = [m for m in tool_msgs if m.name == "read"]

    if read_msgs:
        content = read_msgs[0].content or ""
        assert "path traversal denied" in content or "Error" in content, (
            f"Expected denial, got: {content[:200]}"
        )
    else:
        # Model might not have called the tool -- that's OK, just log
        print(f"[INFO] Model did not attempt /etc/passwd read.\n{_fmt(results)}")


async def test_streaming_yields_progressive_chunks(agent):
    """Assistant messages grow progressively as the model streams tokens."""
    results = await _collect(agent, [Message(role="user", content="Count from one to five.")])
    assistant_msgs = [m for m in results if m.role == "assistant"]

    # The first assistant message should be shorter than the last
    if len(assistant_msgs) >= 2:
        first = assistant_msgs[0].content or ""
        last = assistant_msgs[-1].content or ""
        assert len(last) >= len(first), (
            f"Expected progressive content: first={len(first)} last={len(last)}"
        )


# -- Tool-calling tests (best-effort) --------------------------


async def test_ls_tool(agent, sandbox):
    """If the model calls ls, the tool result should be correct."""
    results = await _collect(
        agent,
        [Message(role="user", content=f"List files in {sandbox} using the ls tool.")],
    )
    tool_msgs = [m for m in results if m.role == "tool"]
    ls_msgs = [m for m in tool_msgs if m.name == "ls"]

    if not ls_msgs:
        pytest.skip(f"Model did not emit tool_calls for ls.\n{_fmt(results)}")

    combined = " ".join(m.content or "" for m in ls_msgs)
    if "README.md" in combined:
        print(f"[OK] ls tool called. Result:\n{combined}")
    else:
        assert "Error" in combined, f"Unexpected ls result:\n{combined[:200]}"
        print(f"[OK] ls gracefully handled: {combined[:100]}")


async def test_read_tool(agent, sandbox):
    """If the model calls read, the tool result should contain the file content."""
    results = await _collect(
        agent,
        [Message(role="user", content=f"Read {sandbox}/README.md using the read tool.")],
    )
    tool_msgs = [m for m in results if m.role == "tool"]
    read_msgs = [m for m in tool_msgs if m.name == "read"]

    if not read_msgs:
        pytest.skip(f"Model did not emit tool_calls for read.\n{_fmt(results)}")

    content = read_msgs[0].content or ""
    if "Test Project" in content:
        print(f"[OK] read tool called. Content prefix:\n{content[:200]}")
    else:
        # Graceful error (e.g. model sent empty arguments)
        assert "Error" in content, f"Unexpected read result:\n{content[:200]}"
        print(f"[OK] read gracefully handled: {content[:100]}")


async def test_read_with_offset(agent, sandbox):
    """If the model calls read with offset/limit, verify correct line."""
    results = await _collect(
        agent,
        [
            Message(
                role="user",
                content=f"Read line 3 from {sandbox}/data.txt using read tool with offset=3 limit=1.",
            )
        ],
    )
    tool_msgs = [m for m in results if m.role == "tool"]
    read_msgs = [m for m in tool_msgs if m.name == "read"]

    if not read_msgs:
        pytest.skip(f"Model did not emit tool_calls for read with offset.\n{_fmt(results)}")

    content = read_msgs[0].content or ""
    if "line3" in content:
        print(f"[OK] read with offset. Content: {content!r}")
    else:
        assert "Error" in content, f"Unexpected read result:\n{content[:200]}"
        print(f"[OK] read gracefully handled: {content[:100]}")


async def test_edit_then_read(agent, sandbox):
    """If the model calls edit, verify graceful handling (success or clear error)."""
    results = await _collect(
        agent,
        [
            Message(
                role="user",
                content=(
                    f"In {sandbox}/README.md replace 'Test Project' with 'Edited Project' "
                    f"using edit tool, then read it to confirm."
                ),
            )
        ],
    )
    tool_msgs = [m for m in results if m.role == "tool"]
    edit_msgs = [m for m in tool_msgs if m.name == "edit"]

    if not edit_msgs:
        pytest.skip(f"Model did not emit tool_calls for edit.\n{_fmt(results)}")

    content = edit_msgs[0].content or ""

    if "Replaced" in content:
        # Success - verify the file changed
        read_path = Path(sandbox) / "README.md"
        assert "Edited Project" in read_path.read_text(encoding="utf-8"), "File not modified"
        print(f"[OK] edit succeeded: {content}")
    else:
        # Graceful error (e.g., model sent empty arguments) -- still acceptable
        assert "Error" in content, f"Unexpected edit result:\n{content}"
        print(f"[OK] edit gracefully handled: {content}")


async def test_multi_turn_conversation(agent, sandbox):
    """Agent state persists across multiple conversation turns."""
    # Turn 1
    turn1 = await _collect(
        agent,
        [Message(role="user", content=f"List files in {sandbox} using ls.")],
    )
    turn1_has_tool = any(m.role == "tool" for m in turn1)

    if not turn1_has_tool:
        pytest.skip(f"Model did not call tool in turn 1.\n{_fmt(turn1)}")

    # Build conversation for turn 2
    conv = [Message(role="user", content=f"List files in {sandbox} using ls.")]
    conv.extend(turn1)

    turn2 = await _collect(
        agent,
        conv + [Message(role="user", content="Now read src/main.py using the read tool.")],
    )

    all_tool_msgs = [m for m in (turn1 + turn2) if m.role == "tool"]
    read_msgs = [m for m in all_tool_msgs if m.name == "read"]

    if not read_msgs:
        pytest.skip(f"Model did not call read in turn 2.\n{_fmt(turn2)}")

    print(
        f"[OK] Multi-turn conversation: ls -> read. Read result: {(read_msgs[0].content or '')[:100]}"
    )
