"""
Root conftest.py for test configuration.

This file sets up the test environment before any imports happen.
"""
import os
import pytest
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up test environment variables before any imports."""
    # Set dummy API keys to prevent import failures
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-unit-tests")
    os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
    os.environ.setdefault("GEMINI_API_KEY", "test-key-for-unit-tests")
    
    # Set test-friendly LLM configurations
    os.environ.setdefault("OPENROUTER_CHAT_MODEL", "test-model")
    os.environ.setdefault("OPENROUTER_BASE_URL", "https://test.example.com")


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(
                    content='{"subgoals": [{"description": "Test subgoal", "type": "subgoal"}], "tasks": [{"description": "Test task", "type": "task"}]}'
                )
            )
        ]
    )
    return mock_client


@pytest.fixture
def mock_instructor_client():
    """Create a mock instructor client for testing."""
    from pydantic import BaseModel
    
    class MockResponse(BaseModel):
        subgoals: list = []
        tasks: list = []
    
    mock_client = Mock()
    mock_client.return_value = MockResponse()
    return mock_client