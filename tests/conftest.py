"""
PyAgent Test Configuration
===========================

Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_llm_response():
    """Fixture providing a mock LLM response."""
    return "This is a mock LLM response for testing purposes."


@pytest.fixture
def sample_agent():
    """Fixture providing a sample agent."""
    from pyagent.easy.agent_factory import agent
    return agent("You are a test agent")


@pytest.fixture
def enable_mock_mode():
    """Fixture that enables mock mode for testing without API calls."""
    from pyagent.easy.config import config
    original = config._mock_mode if hasattr(config, '_mock_mode') else False
    config.enable_mock(True)
    yield
    config.enable_mock(original)
