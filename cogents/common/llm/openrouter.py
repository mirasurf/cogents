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
from typing import TypeVar

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

    def __init__(self, instructor: bool = False):
        """
        Initialize the LLM client.

        Args:
            instructor: Whether to enable instructor for structured output
        """
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

        self.base_url = "https://openrouter.ai/api/v1"

        # Model configurations (can be overridden by environment variables)
        self.chat_model = os.getenv("OPENROUTER_CHAT_MODEL", GEMINI_FLASH)
        self.vision_model = os.getenv("OPENROUTER_VISION_MODEL", GEMINI_FLASH)

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
