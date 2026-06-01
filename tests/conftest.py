"""Shared pytest configuration and fixtures."""

import pytest

from coreloop import clear_registry

# Default base_url and model for tests that construct Agent.
# These require a running Ollama instance with qwen3.5:9b.
TEST_BASE_URL = "http://localhost:11434/v1"
TEST_MODEL = "qwen3.5:9b"


@pytest.fixture(autouse=True)
def _isolate_tool_registry():
    """Keep the global ``@tool`` registry from leaking between tests."""
    clear_registry()
    yield
    clear_registry()
