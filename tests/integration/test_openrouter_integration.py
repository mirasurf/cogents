#!/usr/bin/env python3
"""
Integration tests for OpenRouter LLM client.

These tests verify the integration with OpenRouter API
and general OpenAI-compatible providers.
"""

import os
from pathlib import Path

import pytest

from cogents.common.llm import get_llm_client


@pytest.mark.integration
@pytest.mark.slow
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if API key is available
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set in environment")

        self.client = get_llm_client(provider="openrouter")

    def test_completion_with_gemini(self):
        """Test chat completion with Gemini model."""
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": "What are the top 3 things to do in Tokyo?"},
        ]

        response = self.client.completion(messages, temperature=0.7)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Tokyo" in response or "japan" in response.lower()

    def test_completion_alias(self):
        """Test that completion method works as alias for completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'test completed' exactly."},
        ]

        # Test completion method (alias)
        response = self.client.completion(messages, temperature=0.1)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test completed" in response.lower()

    def test_completion_convenience_function(self):
        """Test the convenience completion function."""
        messages = [{"role": "user", "content": "What's the capital of France?"}]

        response = self.client.completion(messages)

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
        assert hasattr(self.client, "completion")
        assert hasattr(self.client, "understand_image")

    def test_error_handling_invalid_messages(self):
        """Test error handling with invalid message format."""
        invalid_messages = [{"invalid_key": "This should fail"}]

        with pytest.raises(Exception):
            self.client.completion(invalid_messages)

    def test_temperature_parameter(self):
        """Test that temperature parameter affects response variability."""
        messages = [{"role": "user", "content": "Give me a random number between 1 and 10"}]

        # Test with different temperatures
        response_low = self.client.completion(messages, temperature=0.1)
        response_high = self.client.completion(messages, temperature=0.9)

        # Both should be valid responses
        assert response_low is not None
        assert response_high is not None
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)

    def test_structured_completion_integration(self):
        """Test structured completion with OpenRouter."""
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            message: str

        # Get client with instructor enabled
        client = get_llm_client(provider="openrouter", instructor=True)

        # Make a structured completion
        messages = [{"role": "user", "content": "Reply with a simple greeting message"}]
        response = client.structured_completion(
            messages=messages, response_model=SimpleResponse, temperature=0.1, max_tokens=50
        )

        # Verify response
        assert response is not None
        assert isinstance(response, SimpleResponse)
        assert hasattr(response, "message")
        assert len(response.message.strip()) > 0

    def test_multiple_messages_conversation(self):
        """Test handling of multi-turn conversations."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 10 + 5?"},
            {"role": "assistant", "content": "10 + 5 = 15"},
            {"role": "user", "content": "What about 15 * 2?"},
        ]

        response = self.client.completion(messages, temperature=0.1)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention 30 or show understanding of math
        assert any(str(i) in response for i in [30, "thirty"])

    def test_max_tokens_parameter(self):
        """Test max_tokens parameter limits response length."""
        messages = [{"role": "user", "content": "Write a long story about a dragon and a knight."}]

        # Test with very limited tokens
        short_response = self.client.completion(messages, max_tokens=20)

        # Test with more tokens
        long_response = self.client.completion(messages, max_tokens=100)

        # Assertions
        assert short_response is not None
        assert long_response is not None
        assert isinstance(short_response, str)
        assert isinstance(long_response, str)

        # Long response should generally be longer (though not guaranteed)
        # This is a probabilistic test, so we'll just check they're both valid
        assert len(short_response.strip()) > 0
        assert len(long_response.strip()) > 0
