from abc import ABC, abstractmethod
from typing import Any


class SmartVoiceBase(ABC):
    @abstractmethod
    def transcribe(self, audio_data: Any) -> str:
        """Transcribe audio data to text."""
