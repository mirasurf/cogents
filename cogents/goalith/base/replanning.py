from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .update_event import UpdateEvent


class TriggerType(str, Enum):
    """Types of replan triggers."""

    TASK_FAILURE = "task_failure"
    DEADLINE_MISS = "deadline_miss"
    PRIORITY_CHANGE = "priority_change"
    EXTERNAL_SIGNAL = "external_signal"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    DEPENDENCY_CHANGE = "dependency_change"
    MANUAL_TRIGGER = "manual_trigger"


class ReplanScope(str, Enum):
    """Scope of replanning operations."""

    LOCAL = "local"  # Only affected nodes
    SUBTREE = "subtree"  # Node and all descendants
    GLOBAL = "global"  # Entire plan


class ReplanEvent(BaseModel):
    """
    Represents a replanning event.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    trigger_type: TriggerType = Field(..., description="Type of trigger that caused replanning")

    # Event details
    affected_nodes: List[str] = Field(..., description="IDs of nodes that need replanning")
    scope: ReplanScope = Field(..., description="Scope of the replanning operation")
    description: str = Field(default="", description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    source_event: Optional[UpdateEvent] = Field(default=None, description="Original update event that triggered this")

    # Tracking
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow(), description="When the replan event occurred")
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="When the event was triggered"
    )
    handled: bool = Field(default=False, description="Whether this event has been handled")
    executed: bool = Field(default=False, description="Whether this event has been executed")
    execution_time: Optional[float] = Field(default=None, description="Time taken to execute the replan")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result of the replanning operation")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __eq__(self, other: object) -> bool:
        """
        Compare two ReplanEvents for equality based on meaningful content.

        Excludes timestamp field as it's automatically generated.
        """
        if not isinstance(other, ReplanEvent):
            return False

        return (
            self.id == other.id
            and self.trigger_type == other.trigger_type
            and self.affected_nodes == other.affected_nodes
            and self.scope == other.scope
            and self.description == other.description
            and self.metadata == other.metadata
            and self.context == other.context
            and self.source_event == other.source_event
            and self.handled == other.handled
            and self.executed == other.executed
            and self.execution_time == other.execution_time
            and self.result == other.result
            # Exclude triggered_at as it's automatically generated and may differ
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trigger_type": str(self.trigger_type),
            "affected_nodes": self.affected_nodes,
            "scope": str(self.scope),
            "description": self.description,
            "metadata": self.metadata,
            "context": self.context,
            "source_event": self.source_event.to_dict() if self.source_event else None,
            "timestamp": self.timestamp.isoformat(),
            "handled": self.handled,
        }


class ReplanTrigger(ABC):
    """
    Abstract interface for replan triggers.

    Defines conditions that should cause replanning.
    """

    @abstractmethod
    def should_trigger(
        self, event: UpdateEvent, graph_store, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReplanEvent]:
        """
        Check if this trigger should fire for an update event.

        Args:
            event: The update event to check
            graph_store: The graph store for context
            context: Additional context

        Returns:
            ReplanEvent if trigger fires, None otherwise
        """

    @property
    @abstractmethod
    def trigger_type(self) -> TriggerType:
        """Get the type of this trigger."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this trigger."""

    @abstractmethod
    def get_trigger_type(self) -> TriggerType:
        """Get the trigger type."""
