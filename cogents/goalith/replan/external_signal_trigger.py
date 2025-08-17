from typing import Any, Dict, Optional, Set

from ..base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from ..base.update_event import UpdateEvent


class ExternalSignalTrigger(ReplanTrigger):
    """
    Trigger that fires on external signals.
    """

    def __init__(
        self,
        signal_types: Optional[Set[str]] = None,
        name: str = "external_signal_trigger",
    ):
        """
        Initialize external signal trigger.

        Args:
            signal_types: Types of signals to listen for
            name: Name of this trigger
        """
        self._signal_types = signal_types or {
            "user_request",
            "system_update",
            "resource_change",
        }
        self._name = name

    @property
    def trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return TriggerType.EXTERNAL_SIGNAL

    def get_trigger_type(self) -> TriggerType:
        """Get the trigger type."""
        return self.trigger_type

    @property
    def name(self) -> str:
        """Get the trigger name."""
        return self._name

    def get_replan_scope(self, event: UpdateEvent) -> ReplanScope:
        """Get the replan scope for this trigger."""
        return ReplanScope(event.data.get("scope", ReplanScope.LOCAL))

    def should_trigger(
        self, event: UpdateEvent, graph_store, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ReplanEvent]:
        """
        Check if external signal should trigger replanning.

        Args:
            event: The update event
            graph_store: Graph store for context
            context: Additional context

        Returns:
            ReplanEvent if triggered, None otherwise
        """
        signal_type = event.data.get("signal_type")
        if signal_type not in self._signal_types:
            return None

        return ReplanEvent(
            trigger_type=self.trigger_type,
            affected_nodes=event.data.get("affected_nodes", [event.node_id]),
            scope=ReplanScope(event.data.get("scope", ReplanScope.LOCAL)),
            description=f"External signal triggered replan: {signal_type}",
            metadata={
                "signal_type": signal_type,
                "signal_data": event.data.get("signal_data", {}),
            },
            source_event=event,
        )
