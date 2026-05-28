"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import pytest


# Default provider and model for tests that construct Agent.
# These require a running Ollama instance with qwen3.5:9b.
TEST_PROVIDER = "ollama"
TEST_MODEL = "qwen3.5:9b"
