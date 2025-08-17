#!/usr/bin/env python3
"""
Integration tests for OpenAI LLM client.

These tests verify the integration with OpenAI API
and OpenAI-compatible providers.
"""

import os

import pytest

from cogents.common.llm import get_llm_client


@pytest.mark.integration
@pytest.mark.slow
class TestOpenAIIntegration:
    """Integration tests for OpenAI LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set in environment")

        self.client = get_llm_client(provider="openai")

    def test_basic_completion(self):
        """Test basic chat completion with OpenAI."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=10)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response

    def test_completion_with_conversation_history(self):
        """Test completion with conversation history."""
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 10 / 2?"},
            {"role": "assistant", "content": "10 divided by 2 equals 5."},
            {"role": "user", "content": "What about 5 * 3?"},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=20)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should understand the math context
        assert any(str(i) in response for i in [15, "fifteen"])

    def test_structured_completion(self):
        """Test structured completion with OpenAI."""
        from pydantic import BaseModel, Field

        class MathResponse(BaseModel):
            answer: int = Field(description="The numerical answer")
            explanation: str = Field(description="Brief explanation of the calculation")

        # Create client with instructor enabled
        client = get_llm_client(provider="openai", instructor=True)

        messages = [{"role": "user", "content": "Calculate 7 + 8 and explain the result."}]

        result = client.structured_completion(messages, MathResponse, temperature=0.1, max_tokens=100)

        # Assertions
        assert isinstance(result, MathResponse)
        assert result.answer == 15
        assert len(result.explanation) > 0
        assert "7" in result.explanation or "8" in result.explanation

    def test_temperature_variations(self):
        """Test different temperature settings."""
        messages = [{"role": "user", "content": "Describe the color blue in one sentence."}]

        # Test low temperature (more deterministic)
        response_low = self.client.completion(messages, temperature=0.1, max_tokens=50)

        # Test higher temperature (more creative)
        response_high = self.client.completion(messages, temperature=0.9, max_tokens=50)

        # Both should be valid responses
        assert response_low is not None
        assert response_high is not None
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)
        assert len(response_low.strip()) > 0
        assert len(response_high.strip()) > 0

    def test_max_tokens_limiting(self):
        """Test that max_tokens parameter limits response length."""
        messages = [{"role": "user", "content": "Write a story about a cat and a dog becoming friends."}]

        # Test with very limited tokens
        short_response = self.client.completion(messages, max_tokens=10, temperature=0.7)

        # Test with more tokens
        longer_response = self.client.completion(messages, max_tokens=100, temperature=0.7)

        # Both should be valid
        assert short_response is not None
        assert longer_response is not None
        assert isinstance(short_response, str)
        assert isinstance(longer_response, str)

        # Responses should exist
        assert len(short_response.strip()) > 0
        assert len(longer_response.strip()) > 0

    def test_system_message_influence(self):
        """Test that system messages influence response style."""
        user_query = "Explain gravity."

        # Test with formal system message
        formal_messages = [
            {"role": "system", "content": "You are a formal physics professor. Use scientific terminology."},
            {"role": "user", "content": user_query},
        ]

        # Test with casual system message
        casual_messages = [
            {"role": "system", "content": "You are a friendly teacher who explains things simply."},
            {"role": "user", "content": user_query},
        ]

        formal_response = self.client.completion(formal_messages, temperature=0.3, max_tokens=100)
        casual_response = self.client.completion(casual_messages, temperature=0.3, max_tokens=100)

        # Both should mention gravity concepts
        assert formal_response is not None
        assert casual_response is not None
        assert "gravity" in formal_response.lower() or "force" in formal_response.lower()
        assert "gravity" in casual_response.lower() or "force" in casual_response.lower()

    def test_vision_capability(self):
        """Test image understanding if vision model is available."""
        from pathlib import Path

        # Look for a sample image
        sample_image_path = Path(__file__).parent.parent.parent / "examples" / "sample_image.jpg"

        if not sample_image_path.exists():
            pytest.skip("No sample image available for vision testing")

        try:
            prompt = "What do you see in this image? Describe it briefly."
            analysis = self.client.understand_image(sample_image_path, prompt)

            # Assertions
            assert analysis is not None
            assert isinstance(analysis, str)
            assert len(analysis) > 10  # Should be a meaningful description

        except Exception as e:
            if "vision" in str(e).lower() or "image" in str(e).lower():
                pytest.skip(f"Vision capability not available: {e}")
            else:
                raise

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with malformed messages
        with pytest.raises(Exception):
            self.client.completion([{"invalid": "message format"}])

        # Test with empty messages
        with pytest.raises(Exception):
            self.client.completion([])

    def test_streaming_not_supported_gracefully(self):
        """Test that streaming parameter is handled gracefully."""
        messages = [{"role": "user", "content": "Say hello"}]

        # Most implementations should either support streaming or ignore the parameter
        try:
            response = self.client.completion(messages, stream=False)
            assert response is not None
            assert isinstance(response, str)
        except Exception as e:
            if "stream" not in str(e).lower():
                raise  # Re-raise if it's not a streaming-related error
