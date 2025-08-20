from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from .base import BaseLLMClient

T = TypeVar("T")


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instructor: bool = False,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        raise NotImplementedError("Ollama is not supported yet")

    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("Ollama is not supported yet")

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("Ollama is not supported yet")

    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("Ollama is not supported yet")

    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("Ollama is not supported yet")

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Embedding is not supported by the ollama provider")

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        raise NotImplementedError("Embedding is not supported by the ollama provider")

    def rerank(self, query: str, chunks: List[str]) -> List[str]:
        raise NotImplementedError("Reranking is not supported by the ollama provider")
