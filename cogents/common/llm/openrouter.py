"""
LLM utilities for CogentNano using OpenRouter via OpenAI SDK.

This module provides:
- Chat completion using various models via OpenRouter
- Text embeddings using OpenAI text-embedding-3-small
- Image understanding using vision models
- Instructor integration for structured output
- LangSmith tracing for observability
"""

import os
from typing import List, Optional, TypeVar

from cogents.common.consts import GEMINI_FLASH
from cogents.common.llm.openai import LLMClient as OpenAILLMClient

T = TypeVar("T")


class LLMClient(OpenAILLMClient):
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instructor: bool = False,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL for the OpenRouter API (defaults to OpenRouter's URL)
            api_key: API key for authentication (defaults to OPENROUTER_API_KEY env var)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gemini-flash)
            vision_model: Model to use for vision tasks (defaults to gemini-flash)
            **kwargs: Additional arguments to pass to OpenAILLMClient
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
            **kwargs,
        )

        # Configure LangSmith tracing for observability
        self._langsmith_provider = "openrouter"

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's embedding model through OpenRouter.

        Note: OpenRouter doesn't directly support embeddings, so we use OpenAI's
        embedding API directly for this functionality.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            # Use OpenAI directly for embeddings since OpenRouter doesn't support them
            import openai

            # Create a separate OpenAI client for embeddings
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for embeddings. Set OPENAI_API_KEY environment variable.")

            openai_client = openai.OpenAI(api_key=openai_api_key)
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

            response = openai_client.embeddings.create(
                model=embedding_model,
                input=text,
            )

            # Record token usage if available
            try:
                if hasattr(response, "usage") and response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": response.usage.total_tokens,
                        "model_name": embedding_model,
                        "call_type": "embedding",
                    }
                    from cogents.common.llm.token_tracker import TokenUsage

                    usage = TokenUsage(**usage_data)
                    get_token_tracker().record_usage(usage)
                    logger.debug(f"Token usage for embedding: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not track embedding token usage: {e}")

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding with OpenRouter (via OpenAI): {e}")
            raise

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI through OpenRouter.

        Args:
            chunks: List of texts to embed

        Returns:
            List of embedding lists
        """
        try:
            import openai

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for embeddings. Set OPENAI_API_KEY environment variable.")

            openai_client = openai.OpenAI(api_key=openai_api_key)
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

            response = openai_client.embeddings.create(
                model=embedding_model,
                input=chunks,
            )

            # Record token usage if available
            try:
                if hasattr(response, "usage") and response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": response.usage.total_tokens,
                        "model_name": embedding_model,
                        "call_type": "embedding",
                    }
                    from cogents.common.llm.token_tracker import TokenUsage

                    usage = TokenUsage(**usage_data)
                    get_token_tracker().record_usage(usage)
                    logger.debug(f"Token usage for batch embedding: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not track batch embedding token usage: {e}")

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"Error generating batch embeddings with OpenRouter (via OpenAI): {e}")
            # Fallback to individual calls
            embeddings = []
            for chunk in chunks:
                embedding = self.embed(chunk)
                embeddings.append(embedding)
            return embeddings

    def rerank(self, query: str, chunks: List[str]) -> List[str]:
        """
        Rerank chunks based on their relevance to the query using embeddings.

        Uses OpenAI embeddings for similarity calculation since OpenRouter
        doesn't have a native reranking API.

        Args:
            query: The query to rank against
            chunks: List of text chunks to rerank

        Returns:
            Reranked list of chunks
        """
        try:
            # Get embeddings for query and chunks
            query_embedding = self.embed(query)
            chunk_embeddings = self.embed_batch(chunks)

            # Calculate cosine similarity
            import math

            def cosine_similarity(a: List[float], b: List[float]) -> float:
                dot_product = sum(x * y for x, y in zip(a, b))
                magnitude_a = math.sqrt(sum(x * x for x in a))
                magnitude_b = math.sqrt(sum(x * x for x in b))
                if magnitude_a == 0 or magnitude_b == 0:
                    return 0
                return dot_product / (magnitude_a * magnitude_b)

            # Calculate similarities and sort
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i, chunks[i]))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Return reranked chunks
            return [chunk for _, _, chunk in similarities]

        except Exception as e:
            logger.error(f"Error reranking with OpenRouter: {e}")
            # Fallback: return original order
            return chunks
