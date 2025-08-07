from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from .base_delegator import BaseLLMDelegator

T = TypeVar("T")


class LLMClient(BaseLLMDelegator):
    def __init__(self):
        super().__init__()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ):
        pass

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        pass

    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        pass

    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        pass
