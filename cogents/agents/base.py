from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    """Output from research process."""

    content: str = Field(default="")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseResearcher(ABC):
    """Base class for researchers."""

    @abstractmethod
    def research(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> ResearchOutput:
        pass
