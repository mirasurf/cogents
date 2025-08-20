from typing import List

from .base import BaseLLMClient


class LLMClient(BaseLLMClient):
    def __init__(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def completion(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def structured_completion(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def understand_image(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def understand_image_from_url(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def embed_batch(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def embed(self, **kwargs):
        raise NotImplementedError("Litellm is not supported yet")

    def rerank(self, query: str, chunks: List[str]) -> List[str]:
        raise NotImplementedError("Litellm is not supported yet")
