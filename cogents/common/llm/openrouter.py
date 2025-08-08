"""
LLM utilities for CogentNano using OpenRouter via OpenAI SDK.

This module provides:
- Chat completion using various models via OpenRouter
- Text embeddings using OpenAI text-embedding-3-small
- Image understanding using vision models
- Instructor integration for structured output
- LangSmith tracing for observability
"""

import logging
import os
from typing import Optional, TypeVar

from cogents.common.consts import GEMINI_FLASH
from cogents.common.llm.openai import LLMClient as OpenAILLMClient

# Import instructor for structured output

T = TypeVar("T")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class LLMClient(OpenAILLMClient):
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instructor: bool = False,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL for the OpenRouter API (defaults to OpenRouter's URL)
            api_key: API key for authentication (defaults to OPENROUTER_API_KEY env var)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gemini-flash)
            vision_model: Model to use for vision tasks (defaults to gemini-flash)
        """
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide api_key parameter or set OPENROUTER_API_KEY environment variable."
            )

        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Model configurations (can be overridden by environment variables)
        self.chat_model = chat_model or os.getenv("OPENROUTER_CHAT_MODEL", GEMINI_FLASH)
        self.vision_model = vision_model or os.getenv("OPENROUTER_VISION_MODEL", GEMINI_FLASH)

        super().__init__(
            base_url=self.base_url,
            api_key=self.openrouter_api_key,
            instructor=instructor,
            chat_model=self.chat_model,
            vision_model=self.vision_model,
        )

        # Configure LangSmith tracing for observability
        self._langsmith_provider = "openrouter"


# Convenience functions for easy usage
def get_llm_client() -> LLMClient:
    """
    Get an LLM client instance.

    Returns:
        LLMClient instance
    """
    return LLMClient()


def get_llm_client_instructor() -> LLMClient:
    """
    Get an LLM client instance with instructor support.

    Returns:
        LLMClient instance with instructor enabled
    """
    return LLMClient(instructor=True)
