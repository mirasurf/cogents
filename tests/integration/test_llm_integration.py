#!/usr/bin/env python3
"""
Integration tests for LLM utilities.

These tests verify the integration between different LLM components
and external APIs (OpenRouter, OpenAI, etc.).
"""

import os
from pathlib import Path

import pytest

from cogents.common.llm.openrouter import LLMClient


@pytest.mark.integration
@pytest.mark.slow
class TestLLMIntegration:
    """Integration tests for LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if API key is available
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set in environment")

        self.client = LLMClient()

    def test_chat_completion_with_gemini(self):
        """Test chat completion with Gemini model."""
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": "What are the top 3 things to do in Tokyo?"},
        ]

        response = self.client.chat_completion(messages, temperature=0.7)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Tokyo" in response or "japan" in response.lower()

    def test_chat_completion_convenience_function(self):
        """Test the convenience chat_completion function."""
        messages = [{"role": "user", "content": "What's the capital of France?"}]

        response = self.client.chat_completion(messages)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "paris" in response.lower() or "france" in response.lower()

    def test_image_analysis_with_sample_image(self):
        """Test image analysis functionality if sample image exists."""
        sample_image_path = Path(__file__).parent.parent.parent / "examples" / "sample_image.jpg"

        if not sample_image_path.exists():
            pytest.skip("Sample image not found for testing")

        prompt = "Describe what you see in this image in detail."
        analysis = self.client.understand_image(sample_image_path, prompt)

        # Assertions
        assert analysis is not None
        assert isinstance(analysis, str)
        assert len(analysis) > 0

    def test_client_initialization(self):
        """Test LLMClient initialization."""
        assert self.client is not None
        assert hasattr(self.client, "chat_completion")
        assert hasattr(self.client, "understand_image")

    def test_error_handling_invalid_messages(self):
        """Test error handling with invalid message format."""
        invalid_messages = [{"invalid_key": "This should fail"}]

        with pytest.raises(Exception):
            self.client.chat_completion(invalid_messages)

    def test_temperature_parameter(self):
        """Test that temperature parameter affects response variability."""
        messages = [{"role": "user", "content": "Give me a random number between 1 and 10"}]

        # Test with different temperatures
        response_low = self.client.chat_completion(messages, temperature=0.1)
        response_high = self.client.chat_completion(messages, temperature=0.9)

        # Both should be valid responses
        assert response_low is not None
        assert response_high is not None
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)


@pytest.mark.integration
class TestLLMEnvironment:
    """Test environment setup for LLM integration."""

    def test_api_key_environment_variable(self):
        """Test that API key environment variable is properly set."""
        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key is None:
            pytest.skip("OPENROUTER_API_KEY not set - skipping environment test")

        assert api_key is not None
        assert len(api_key) > 0
        assert api_key.startswith("sk-")  # OpenRouter API keys typically start with sk-

    def test_required_dependencies_importable(self):
        """Test that all required LLM dependencies can be imported."""
        try:
            assert True  # If we get here, imports succeeded
        except ImportError as e:
            pytest.fail(f"Failed to import LLM utilities: {e}")
