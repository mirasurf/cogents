from typing import Any, Dict, Optional

from ..base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from ..base.update_event import UpdateEvent, UpdateType


class PriorityChangeTrigger(ReplanTrigger):
    """
    Trigger that fires when significant priority changes occur.
    """

    def __init__(self, threshold: float = 2.0, name: str = "priority_change_trigger"):
        """
        Initialize priority change trigger.

        Args:
            threshold: Minimum priority change to trigger replan
            name: Name of this trigger
        """
        self._threshold = threshold
        self._name = name

    @property
    def trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return TriggerType.PRIORITY_CHANGE

    def get_trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return self.trigger_type

    @property
    def name(self) -> str:
        """Get the trigger name."""
        return self._name

    @property
    def threshold(self) -> float:
        """Get the threshold."""
        return self._threshold

    def should_trigger(
        self, event: UpdateEvent, graph_store, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReplanEvent]:
        """
        Check if priority change should trigger replanning.

        Args:
            event: The update event
            graph_store: Graph store for context
            context: Additional context

        Returns:
            ReplanEvent if triggered, None otherwise
        """
        if event.update_type != UpdateType.PRIORITY_CHANGE:
            return None

        new_priority = event.data.get("new_priority")
        if new_priority is None:
            return None

        if not graph_store.has_node(event.node_id):
            return None

        node = graph_store.get_node(event.node_id)
        priority_change = abs(new_priority - node.priority)

        if priority_change >= self._threshold:
            return ReplanEvent(
                trigger_type=self.trigger_type,
                affected_nodes=[event.node_id],
                scope=ReplanScope.LOCAL,
                description=f"Significant priority change for node {event.node_id}: {priority_change}",
                metadata={
                    "old_priority": node.priority,
                    "new_priority": new_priority,
                    "change_magnitude": priority_change,
                    "threshold": self._threshold,
                },
                source_event=event,
            )

        return None
