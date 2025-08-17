from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..base.goal_node import NodeStatus
from ..base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from ..base.update_event import UpdateEvent


class DeadlineTrigger(ReplanTrigger):
    """
    Trigger that fires when deadlines are missed or approaching.
    """

    def __init__(
        self,
        warning_threshold: timedelta = timedelta(hours=1),
        name: str = "deadline_trigger",
        threshold: Optional[timedelta] = None,
    ):
        """
        Initialize deadline trigger.

        Args:
            warning_threshold: How far in advance to trigger warnings
            name: Name of this trigger
            threshold: Alias for warning_threshold for backward compatibility
        """
        self._warning_threshold = threshold or warning_threshold
        self._name = name

    @property
    def trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return TriggerType.DEADLINE_MISS

    def get_trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return self.trigger_type

    @property
    def name(self) -> str:
        """Get the trigger name."""
        return self._name

    @property
    def threshold(self) -> timedelta:
        """Get the threshold (alias for warning_threshold)."""
        return self._warning_threshold

    @property
    def warning_threshold(self) -> timedelta:
        """Get the warning threshold."""
        return self._warning_threshold

    def should_trigger(
        self, event: UpdateEvent, graph_store, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReplanEvent]:
        """
        Check if deadline issues should trigger replanning.

        Args:
            event: The update event
            graph_store: Graph store for context
            context: Additional context

        Returns:
            ReplanEvent if triggered, None otherwise
        """
        # Check all nodes for deadline issues
        current_time = datetime.now(timezone.utc)
        at_risk_nodes = []

        for node in graph_store.get_all_nodes():
            if not node.deadline:
                continue

            if node.status in {NodeStatus.COMPLETED, NodeStatus.CANCELLED}:
                continue

            time_to_deadline = node.deadline - current_time

            # Check if deadline is missed or approaching
            if time_to_deadline < timedelta(0):
                # Deadline missed
                at_risk_nodes.append((node.id, "missed"))
            elif time_to_deadline < self._warning_threshold:
                # Deadline approaching
                at_risk_nodes.append((node.id, "approaching"))

        if at_risk_nodes:
            return ReplanEvent(
                trigger_type=self.trigger_type,
                affected_nodes=[node_id for node_id, _ in at_risk_nodes],
                scope=ReplanScope.GLOBAL,
                description=f"Deadline issues detected for {len(at_risk_nodes)} nodes",
                metadata={
                    "at_risk_nodes": at_risk_nodes,
                    "warning_threshold": self._warning_threshold.total_seconds(),
                },
                source_event=event,
            )

        return None
