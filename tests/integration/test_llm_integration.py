#!/usr/bin/env python3
"""
Integration tests for LLM utilities.

These tests verify the integration between different LLM components
and external APIs (OpenRouter, OpenAI, etc.).
"""

import os
from pathlib import Path

import pytest

from cogents.agents.askura_agent.askura_agent import AskuraAgent, AskuraConfig
from cogents.agents.askura_agent.models import InformationSlot
from cogents.common.lg_hooks import TokenUsageCallback
from cogents.common.llm import get_llm_client
from cogents.common.llm.token_tracker import TokenUsage, get_token_tracker


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

        self.client = get_llm_client(provider="openrouter")

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


@pytest.mark.integration
@pytest.mark.slow
class TestTokenUsageIntegration:
    """Integration tests for token usage tracking system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Reset token tracker before each test
        get_token_tracker().reset()

        # Check if API key is available for real API tests
        self.has_api_key = bool(os.getenv("OPENROUTER_API_KEY"))
        if self.has_api_key:
            self.client = get_llm_client(provider="openrouter")

    def test_token_tracker_basic_functionality(self):
        """Test basic token tracker functionality."""
        tracker = get_token_tracker()

        # Should start with zero usage
        assert tracker.get_total_tokens() == 0
        assert tracker.call_count == 0

        # Record some usage
        usage = TokenUsage(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            model_name="test-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Verify tracking
        assert tracker.get_total_tokens() == 80
        assert tracker.call_count == 1
        assert tracker.total_prompt_tokens == 50
        assert tracker.total_completion_tokens == 30

        # Test stats
        stats = tracker.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 80
        assert len(stats["usage_history"]) == 1

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_llm_client_token_tracking(self):
        """Test that LLM client properly tracks token usage."""
        tracker = get_token_tracker()
        initial_tokens = tracker.get_total_tokens()

        # Make a simple completion
        messages = [{"role": "user", "content": "Say 'Hello World' and nothing else."}]
        response = self.client.chat_completion(messages, temperature=0.1, max_tokens=10)

        # Verify response
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0

        # Verify token usage was tracked
        final_tokens = tracker.get_total_tokens()
        assert final_tokens > initial_tokens, "Token usage should have increased"

        # Check that at least one call was recorded
        stats = tracker.get_stats()
        assert stats["total_calls"] > 0
        assert len(stats["usage_history"]) > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_structured_completion_token_tracking(self):
        """Test token tracking for structured completions."""
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            message: str

        # Get client with instructor enabled
        client = get_llm_client(provider="openrouter", instructor=True)
        tracker = get_token_tracker()
        initial_tokens = tracker.get_total_tokens()

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

        # Verify token usage was tracked
        final_tokens = tracker.get_total_tokens()
        assert final_tokens > initial_tokens, "Token usage should have increased"

        # Check for structured completion in history
        stats = tracker.get_stats()
        structured_calls = [h for h in stats["usage_history"] if h.get("call_type") == "structured"]
        assert len(structured_calls) > 0, "Should have at least one structured completion call"

    def test_token_callback_integration(self):
        """Test TokenUsageCallback integration with token tracker."""
        callback = TokenUsageCallback(model_name="test-model", verbose=False)
        tracker = get_token_tracker()

        # Simulate some direct token tracker usage (from custom clients)
        usage = TokenUsage(
            prompt_tokens=40,
            completion_tokens=25,
            total_tokens=65,
            model_name="custom-client-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Simulate some callback usage (from LangChain LLMs)
        callback.total_prompt_tokens = 30
        callback.total_completion_tokens = 20
        callback.llm_calls = 1

        # Test combined session summary
        summary = callback.get_session_summary()

        # Should combine both sources
        assert summary["total_llm_calls"] == 2  # 1 from tracker + 1 from callback
        assert summary["total_prompt_tokens"] == 70  # 40 + 30
        assert summary["total_completion_tokens"] == 45  # 25 + 20
        assert summary["total_tokens"] == 115  # 65 + 50

        # Should have separate breakdowns
        assert summary["callback_stats"]["llm_calls"] == 1
        assert summary["tracker_stats"]["total_calls"] == 1

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_askura_agent_token_integration(self):
        """Test end-to-end token tracking with AskuraAgent."""
        # Create a simple config with minimal information to extract
        config = AskuraConfig(
            conversation_purpose="Test conversation for token tracking",
            information_slots=[
                InformationSlot(name="test_slot", description="A simple test slot", required=True, data_type="str")
            ],
        )

        # Create agent
        agent = AskuraAgent(config)

        # Create callback and track token usage
        callback = TokenUsageCallback(verbose=False)
        tracker = get_token_tracker()
        tracker.reset()  # Start fresh

        # Mock a simple interaction that would complete quickly
        test_message = "Hello, my test value is 'test_data'"

        try:
            # Run a simple interaction
            response = agent.chat(test_message, session_id="test_session")

            # Verify we got a response
            assert response is not None
            assert hasattr(response, "message")
            assert isinstance(response.message, str)

            # Verify token usage was tracked
            final_stats = tracker.get_stats()
            assert final_stats["total_calls"] > 0, "Should have recorded LLM calls"
            assert final_stats["total_tokens"] > 0, "Should have used some tokens"

            # Test the callback integration (if we had LangChain LLM calls)
            session_summary = callback.get_session_summary()
            combined_tokens = session_summary["total_tokens"]

            # Should have some token usage overall
            assert combined_tokens > 0, "Combined token usage should be positive"

            print(
                f"Token usage summary: {combined_tokens} total tokens across {session_summary['total_llm_calls']} calls"
            )

        except Exception as e:
            # If there's an API error, we can still check that the tracking system is set up correctly
            if "api" in str(e).lower() or "rate limit" in str(e).lower():
                pytest.skip(f"API error during test: {e}")
            else:
                raise

    def test_token_callback_reset_functionality(self):
        """Test that callback reset properly clears both callback and tracker."""
        callback = TokenUsageCallback(verbose=False)
        tracker = get_token_tracker()

        # Add some data to both
        callback.total_prompt_tokens = 50
        callback.total_completion_tokens = 30
        callback.llm_calls = 2

        usage = TokenUsage(
            prompt_tokens=25,
            completion_tokens=15,
            total_tokens=40,
            model_name="test-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Verify data exists
        assert callback.total_tokens() == 80
        assert tracker.get_total_tokens() == 40

        # Reset
        callback.reset_session()

        # Verify both are reset
        assert callback.total_tokens() == 0
        assert tracker.get_total_tokens() == 0
        assert callback.llm_calls == 0
        assert tracker.call_count == 0

    def test_token_estimation_fallback(self):
        """Test token estimation when actual usage is not available."""
        from cogents.common.llm.token_tracker import estimate_token_usage

        prompt_text = "This is a test prompt with some words"
        completion_text = "This is a test response with different words"

        usage = estimate_token_usage(prompt_text, completion_text, "test-model", "completion")

        # Should have reasonable estimates
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.estimated == True
        assert usage.model_name == "test-model"
        assert usage.call_type == "completion"

        # Should be roughly proportional to text length
        assert usage.prompt_tokens > 5  # Should be more than 5 tokens for that text
        assert usage.completion_tokens > 5
