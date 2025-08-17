from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class UpdateType(str, Enum):
    """Types of update events."""

    STATUS_CHANGE = "status_change"
    NODE_EDIT = "node_edit"
    NODE_ADD = "node_add"
    NODE_REMOVE = "node_remove"
    DEPENDENCY_ADD = "dependency_add"
    DEPENDENCY_REMOVE = "dependency_remove"
    CONTEXT_UPDATE = "context_update"
    PRIORITY_CHANGE = "priority_change"


class UpdateEvent(BaseModel):
    """
    Represents an update event in the system.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    update_type: UpdateType = Field(..., description="Type of update")
    node_id: str = Field(..., description="ID of the affected node")

    # Event data
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional update data")
    source: Optional[str] = Field(default=None, description="Source of the update (agent, user, etc.)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the update occurred"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __eq__(self, other: object) -> bool:
        """
        Compare two UpdateEvents for equality based on meaningful content.

        Excludes timestamp field as it's automatically generated and not part of the logical state.
        """
        if not isinstance(other, UpdateEvent):
            return False

        return (
            self.id == other.id
            and self.update_type == other.update_type
            and self.node_id == other.node_id
            and self.data == other.data
            and self.source == other.source
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "update_type": str(self.update_type),
            "node_id": self.node_id,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateEvent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            update_type=UpdateType(data["update_type"]),
            node_id=data["node_id"],
            data=data.get("data", {}),
            source=data.get("source"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(timezone.utc),
        )
