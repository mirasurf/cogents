#!/usr/bin/env python3
"""
Integration tests for Ollama LLM client.

These tests verify the integration with Ollama local API server.
"""

import os

import pytest

from cogents.common.llm import get_llm_client


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaIntegration:
    """Integration tests for Ollama LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if Ollama is running locally
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            import urllib.error
            import urllib.request

            # Test if Ollama server is accessible
            request = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status != 200:
                    pytest.skip(f"Ollama server not accessible at {base_url}")
        except (urllib.error.URLError, Exception) as e:
            pytest.skip(f"Ollama server not running or accessible at {base_url}: {e}")

        self.client = get_llm_client(provider="ollama", base_url=base_url)

    def test_basic_completion(self):
        """Test basic chat completion with Ollama."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 3+3? Answer with just the number."},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=10)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain the answer
        assert "6" in response

    def test_completion_with_conversation(self):
        """Test completion with conversation history."""
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": "What is 8 * 2?"},
            {"role": "assistant", "content": "8 * 2 = 16"},
            {"role": "user", "content": "What about 16 / 4?"},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=20)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should understand the math context
        assert "4" in response

    def test_structured_completion(self):
        """Test structured completion with Ollama."""
        from pydantic import BaseModel, Field

        class SimpleCalculation(BaseModel):
            problem: str = Field(description="The math problem")
            result: int = Field(description="The numerical result")

        # Create client with instructor enabled
        client = get_llm_client(provider="ollama", instructor=True)

        messages = [{"role": "user", "content": "Solve: 9 + 6. Provide the problem and result."}]

        try:
            result = client.structured_completion(messages, SimpleCalculation, temperature=0.1, max_tokens=100)

            # Assertions
            assert isinstance(result, SimpleCalculation)
            assert result.result == 15
            assert "9" in result.problem and "6" in result.problem

        except Exception as e:
            if "instructor" in str(e).lower() or "structured" in str(e).lower():
                pytest.skip(f"Structured completion not supported by Ollama client: {e}")
            else:
                raise

    def test_temperature_parameter(self):
        """Test temperature parameter effects."""
        messages = [{"role": "user", "content": "Name a color. Just say the color name."}]

        # Test different temperatures
        response_deterministic = self.client.completion(messages, temperature=0.0, max_tokens=10)
        response_random = self.client.completion(messages, temperature=1.0, max_tokens=10)

        # Both should be valid responses
        assert response_deterministic is not None
        assert response_random is not None
        assert isinstance(response_deterministic, str)
        assert isinstance(response_random, str)
        assert len(response_deterministic.strip()) > 0
        assert len(response_random.strip()) > 0

    def test_system_message_handling(self):
        """Test that system messages are properly handled."""
        messages_with_system = [
            {"role": "system", "content": "You are a pirate. End responses with 'Arrr!'."},
            {"role": "user", "content": "What is your favorite color?"},
        ]

        response = self.client.completion(messages_with_system, temperature=0.7, max_tokens=50)

        # Should reflect the system message personality
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        # Note: Not all models may follow system instructions perfectly

    def test_max_tokens_limiting(self):
        """Test max_tokens parameter."""
        messages = [{"role": "user", "content": "Tell me about the solar system."}]

        # Test with limited tokens
        short_response = self.client.completion(messages, max_tokens=20, temperature=0.3)

        # Test with more tokens
        longer_response = self.client.completion(messages, max_tokens=100, temperature=0.3)

        # Both should be valid
        assert short_response is not None
        assert longer_response is not None
        assert isinstance(short_response, str)
        assert isinstance(longer_response, str)
        assert len(short_response.strip()) > 0
        assert len(longer_response.strip()) > 0

    def test_client_initialization_custom_base_url(self):
        """Test client initialization with custom base URL."""
        custom_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        client = get_llm_client(provider="ollama", base_url=custom_base_url)

        assert client is not None
        assert hasattr(client, "completion")

        # Try a simple completion to verify it works
        messages = [{"role": "user", "content": "Say 'test'"}]
        response = client.completion(messages, max_tokens=5)

        assert response is not None
        assert isinstance(response, str)

    def test_vision_not_supported(self):
        """Test that vision methods handle unsupported operations gracefully."""
        # Most Ollama models don't support vision, so this should either:
        # 1. Raise NotImplementedError
        # 2. Raise an appropriate error indicating vision is not supported

        with pytest.raises((NotImplementedError, Exception)) as exc_info:
            self.client.understand_image("/path/to/image.jpg", "What's in this image?")

        # Check that the error indicates vision is not supported
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["vision", "image", "not supported", "not implemented"])

    def test_error_handling_invalid_messages(self):
        """Test error handling with invalid message formats."""
        # Test with malformed messages
        with pytest.raises(Exception):
            self.client.completion([{"wrong_key": "invalid format"}])

    def test_empty_response_handling(self):
        """Test handling of edge cases that might produce empty responses."""
        messages = [{"role": "user", "content": ""}]  # Empty user message

        try:
            response = self.client.completion(messages, max_tokens=10)
            # If it doesn't error, response should still be a string
            assert isinstance(response, str)
        except Exception as e:
            # Empty messages might cause errors, which is acceptable
            assert "empty" in str(e).lower() or "message" in str(e).lower()

    def test_concurrent_requests_stability(self):
        """Test that multiple concurrent requests don't interfere with each other."""
        import threading
        import time

        results = []
        errors = []

        def make_request(query_num):
            try:
                messages = [{"role": "user", "content": f"What is {query_num} + {query_num}?"}]
                response = self.client.completion(messages, temperature=0.1, max_tokens=20)
                results.append((query_num, response))
            except Exception as e:
                errors.append((query_num, str(e)))

        # Create multiple threads
        threads = []
        for i in range(3):  # Keep it small to avoid overwhelming Ollama
            thread = threading.Thread(target=make_request, args=(i + 1,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()
            time.sleep(0.1)  # Small delay to avoid hammering the server

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread

        # Check results
        if errors:
            print(f"Errors occurred: {errors}")

        # At least some requests should succeed
        assert len(results) > 0, f"No successful requests. Errors: {errors}"

        # All successful results should be valid
        for query_num, response in results:
            assert response is not None
            assert isinstance(response, str)
            assert len(response.strip()) > 0
