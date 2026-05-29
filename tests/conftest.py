"""Shared pytest configuration and fixtures."""

import pytest

from minimal_agent import clear_registry

# Default provider and model for tests that construct Agent.
# These require a running Ollama instance with qwen3.5:9b.
TEST_PROVIDER = "ollama"
TEST_MODEL = "qwen3.5:9b"


@pytest.fixture(autouse=True)
def _isolate_tool_registry():
    """Keep the global ``@tool`` registry from leaking between tests."""
    clear_registry()
    yield
    clear_registry()
