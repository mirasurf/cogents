"""
Shared models for vector store implementations.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class OutputData(BaseModel):
    """Standard output data structure for vector store operations."""

    id: str
    score: Optional[float] = None
    payload: Dict[str, Any]
