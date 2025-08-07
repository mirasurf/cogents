"""
LLM utilities for CogentNano using OpenAI-compatible APIs.

This module provides:
- Chat completion using various models via OpenAI-compatible endpoints
- Image understanding using vision models
- Instructor integration for structured output
- LangSmith tracing for observability
- Configurable base URL and API key for OpenAI-compatible services
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

# Import instructor for structured output
from instructor import Instructor, Mode, patch
from openai import OpenAI

from cogents.common.langsmith import configure_langsmith, is_langsmith_enabled
from cogents.common.llm.base import BaseLLMClient

T = TypeVar("T")

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class LLMClient(BaseLLMClient):
    """Client for interacting with OpenAI-compatible LLM services."""

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
            base_url: Base URL for the OpenAI-compatible API (defaults to OpenAI's URL)
            api_key: API key for authentication (defaults to OPENAI_API_KEY env var)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gpt-3.5-turbo)
            vision_model: Model to use for vision tasks (defaults to gpt-4-vision-preview)
        """
        # Configure LangSmith tracing for observability
        configure_langsmith()
        self._langsmith_provider = "openai"

        # Set API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Provide api_key parameter or set OPENAI_API_KEY environment variable."
            )

        # Set base URL (defaults to OpenAI if not provided)
        self.base_url = base_url

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

        # Model configurations
        self.chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL", "gpt-4-vision-preview")

        # Initialize instructor if requested
        self.instructor = None
        if instructor:
            # Create instructor instance for structured output
            patched_client = patch(self.client, mode=Mode.JSON)
            self.instructor = Instructor(
                client=patched_client,
                create=patched_client.chat.completions.create,
                mode=Mode.JSON,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate chat completion using the configured model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to OpenAI API

        Returns:
            Generated text response or streaming response
        """
        try:
            # Add LangSmith metadata for tracing
            if is_langsmith_enabled():
                # Add model information to kwargs for better tracing
                kwargs.setdefault("extra_headers", {})
                kwargs["extra_headers"]["X-LangSmith-Model"] = self.chat_model
                kwargs["extra_headers"]["X-LangSmith-Provider"] = self._langsmith_provider

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        """
        Generate structured completion using instructor.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to instructor

        Returns:
            Structured response as the specified model type
        """
        if not self.instructor:
            raise ValueError("Instructor is not enabled. Initialize LLMClient with instructor=True")

        try:
            # Add LangSmith metadata for structured completion tracing
            if is_langsmith_enabled():
                kwargs.setdefault("extra_headers", {})
                kwargs["extra_headers"]["X-LangSmith-Model"] = self.chat_model
                kwargs["extra_headers"]["X-LangSmith-Provider"] = self._langsmith_provider
                kwargs["extra_headers"]["X-LangSmith-Type"] = "structured"
                kwargs["extra_headers"]["X-LangSmith-Response-Model"] = response_model.__name__

            result = self.instructor.create(
                model=self.chat_model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return result

        except Exception as e:
            logger.error(f"Error in structured completion: {e}")
            raise

    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image using the configured vision model.

        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """
        try:
            # Read and encode the image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]

            # Add LangSmith metadata for vision completion tracing
            if is_langsmith_enabled():
                kwargs.setdefault("extra_headers", {})
                kwargs["extra_headers"]["X-LangSmith-Model"] = self.vision_model
                kwargs["extra_headers"]["X-LangSmith-Provider"] = self._langsmith_provider
                kwargs["extra_headers"]["X-LangSmith-Type"] = "vision"

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image from URL using the configured vision model.

        Args:
            image_url: URL of the image
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

            # Add LangSmith metadata for vision completion tracing
            if is_langsmith_enabled():
                kwargs.setdefault("extra_headers", {})
                kwargs["extra_headers"]["X-LangSmith-Model"] = self.vision_model
                kwargs["extra_headers"]["X-LangSmith-Provider"] = self._langsmith_provider
                kwargs["extra_headers"]["X-LangSmith-Type"] = "vision-url"

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image from URL: {e}")
            raise
