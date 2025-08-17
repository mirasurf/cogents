from typing import Any, Dict, List, Optional

from ..base.goal_node import NodeStatus
from ..base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from ..base.update_event import UpdateEvent, UpdateType


class TaskFailureTrigger(ReplanTrigger):
    """
    Trigger that fires when tasks fail.
    """

    def __init__(self, max_retries: int = 3, name: str = "task_failure_trigger"):
        """
        Initialize task failure trigger.

        Args:
            max_retries: Maximum retries before triggering replan
            name: Name of this trigger
        """
        self._max_retries = max_retries
        self._name = name

    @property
    def trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return TriggerType.TASK_FAILURE

    def get_trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return self.trigger_type

    @property
    def name(self) -> str:
        """Get the trigger name."""
        return self._name

    @property
    def max_retries(self) -> int:
        """Get the maximum retries."""
        return self._max_retries

    def get_affected_nodes(self, event: UpdateEvent) -> List[str]:
        """Get nodes affected by this trigger."""
        return [event.node_id]

    def get_replan_scope(self, event: UpdateEvent) -> ReplanScope:
        """Get the replan scope for this trigger."""
        return ReplanScope.SUBTREE

    def should_trigger(
        self, event: UpdateEvent, graph_store, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReplanEvent]:
        """
        Check if task failure should trigger replanning.

        Args:
            event: The update event
            graph_store: Graph store for context
            context: Additional context

        Returns:
            ReplanEvent if triggered, None otherwise
        """
        if event.update_type != UpdateType.STATUS_CHANGE:
            return None

        new_status = event.data.get("new_status")
        if new_status != NodeStatus.FAILED:
            return None

        if not graph_store.has_node(event.node_id):
            return None

        node = graph_store.get_node(event.node_id)

        # Check if we've exceeded retry limit
        if node.retry_count >= self._max_retries:
            return ReplanEvent(
                trigger_type=self.trigger_type,
                affected_nodes=[event.node_id],
                scope=ReplanScope.SUBTREE,
                description=f"Task {event.node_id} failed {node.retry_count} times, triggering replan",
                metadata={
                    "retry_count": node.retry_count,
                    "max_retries": self._max_retries,
                    "error_message": event.data.get("error_message"),
                },
                source_event=event,
            )

        return None
