from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    """Output from research process."""

    content: str = Field(default="")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
