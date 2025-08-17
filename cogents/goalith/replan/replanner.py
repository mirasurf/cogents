from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base.goal_node import GoalNode, NodeStatus, NodeType
from ..base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from ..base.update_event import UpdateEvent


class Replanner:
    """
    Handles replanning operations when triggers fire.

    Hooks into decomposition and scheduling modules to adjust plans.
    """

    def __init__(self, graph_store, decomposer_registry, scheduler):
        """
        Initialize replanner.

        Args:
            graph_store: The graph store
            decomposer_registry: Registry for decomposers
            scheduler: Scheduler for prioritization
        """
        self._graph_store = graph_store
        self._decomposer_registry = decomposer_registry
        self._scheduler = scheduler
        self._triggers: List[ReplanTrigger] = []
        self._replanning_stats = {
            "total_events": 0,
            "triggers_fired": 0,
            "replans_executed": 0,
            "by_trigger_type": {},
        }

    @property
    def graph_store(self):
        """Get the graph store."""
        return self._graph_store

    @property
    def decomposer_registry(self):
        """Get the decomposer registry."""
        return self._decomposer_registry

    @property
    def scheduler(self):
        """Get the scheduler."""
        return self._scheduler

    @property
    def triggers(self) -> List[ReplanTrigger]:
        """Get the list of triggers."""
        return self._triggers.copy()

    def add_trigger(self, trigger: ReplanTrigger) -> None:
        """
        Add a replan trigger.

        Args:
            trigger: The trigger to add
        """
        self._triggers.append(trigger)

    def remove_trigger(self, trigger) -> None:
        """
        Remove a replan trigger by name or object.

        Args:
            trigger: Trigger object or trigger name to remove
        """
        if isinstance(trigger, str):
            # Remove by name
            self._triggers = [t for t in self._triggers if t.name != trigger]
        else:
            # Remove by object
            self._triggers = [t for t in self._triggers if t is not trigger]

    def check_triggers(self, event: UpdateEvent, context: Optional[Dict[str, Any]] = None) -> List[ReplanEvent]:
        """
        Check all triggers for an update event.

        Args:
            event: The update event
            context: Additional context

        Returns:
            List of replan events that were triggered
        """
        self._replanning_stats["total_events"] += 1
        replan_events = []

        for trigger in self._triggers:
            replan_event = trigger.should_trigger(event, self._graph_store, context)
            if replan_event:
                replan_events.append(replan_event)
                self._replanning_stats["triggers_fired"] += 1

                trigger_type = str(trigger.trigger_type)
                self._replanning_stats["by_trigger_type"][trigger_type] = (
                    self._replanning_stats["by_trigger_type"].get(trigger_type, 0) + 1
                )

        return replan_events

    def execute_replan(self, replan_event: ReplanEvent) -> bool:
        """
        Execute a replanning operation.

        Args:
            replan_event: The replan event to handle

        Returns:
            True if replanning was successful, False otherwise
        """
        try:
            self._replanning_stats["replans_executed"] += 1

            # Handle scope by expanding affected nodes if needed
            affected_nodes = list(replan_event.affected_nodes)
            if replan_event.scope == ReplanScope.SUBTREE:
                # Expand to include all descendants for subtree scope
                expanded_nodes = set(affected_nodes)
                for node_id in replan_event.affected_nodes:
                    if self._graph_store.has_node(node_id):
                        descendants = self._graph_store.get_descendants(node_id)
                        expanded_nodes.update(descendants)
                affected_nodes = list(expanded_nodes)
            elif replan_event.scope == ReplanScope.GLOBAL:
                # For global scope, include all nodes
                all_nodes = self._graph_store.get_all_nodes()
                if isinstance(all_nodes, dict):
                    affected_nodes = list(all_nodes.keys())
                else:
                    # If it's a list, assume it's already node IDs
                    affected_nodes = list(all_nodes)

            # Update the replan event with expanded nodes
            replan_event.affected_nodes = affected_nodes

            if replan_event.trigger_type == TriggerType.TASK_FAILURE:
                return self._handle_task_failure(replan_event)
            elif replan_event.trigger_type == TriggerType.DEADLINE_MISS:
                return self._handle_deadline_miss(replan_event)
            elif replan_event.trigger_type == TriggerType.PRIORITY_CHANGE:
                return self._handle_priority_change(replan_event)
            elif replan_event.trigger_type == TriggerType.EXTERNAL_SIGNAL:
                return self._handle_external_signal(replan_event)
            else:
                return self._handle_generic_replan(replan_event)

        except Exception as e:
            print(f"Error executing replan {replan_event.id}: {e}")
            return False
        finally:
            replan_event.handled = True

    def _handle_task_failure(self, replan_event: ReplanEvent) -> bool:
        """Handle task failure replanning."""
        for node_id in replan_event.affected_nodes:
            if not self._graph_store.has_node(node_id):
                continue

            node = self._graph_store.get_node(node_id)

            # Try alternative decomposition if available
            if node.type == NodeType.GOAL and node.decomposer_name:
                # Re-decompose with different strategy
                if self._decomposer_registry.has_decomposer("fallback_decomposer"):
                    try:
                        # Remove failed children
                        for child_id in list(node.children):
                            if self._graph_store.has_node(child_id):
                                self._graph_store.remove_node(child_id)

                        # Re-decompose
                        new_children = self._decomposer_registry.decompose("fallback_decomposer", node)

                        # Add new children to graph
                        for child in new_children:
                            self._graph_store.add_node(child)
                            self._graph_store.add_parent_child(node_id, child.id)

                        # Reset node status
                        node.status = NodeStatus.PENDING
                        node.retry_count = 0
                        self._graph_store.update_node(node)

                        return True

                    except Exception as e:
                        print(f"Failed to re-decompose node {node_id}: {e}")

            # Fallback: mark as cancelled and try to continue
            node.mark_cancelled()
            self._graph_store.update_node(node)

        return True

    def _handle_deadline_miss(self, replan_event: ReplanEvent) -> bool:
        """Handle deadline miss replanning."""
        # Reprioritize based on deadlines
        for node_id in replan_event.affected_nodes:
            if not self._graph_store.has_node(node_id):
                continue

            node = self._graph_store.get_node(node_id)

            # Boost priority for urgent items
            if node.deadline and node.deadline < datetime.now(timezone.utc):
                node.priority += 10.0  # Boost priority significantly
            elif node.deadline:
                # Boost based on urgency
                time_to_deadline = node.deadline - datetime.now(timezone.utc)
                urgency_boost = max(1.0, 5.0 / (time_to_deadline.total_seconds() / 3600 + 1))
                node.priority += urgency_boost

            self._graph_store.update_node(node)

        return True

    def _handle_priority_change(self, replan_event: ReplanEvent) -> bool:
        """Handle priority change replanning."""
        # Update scheduling order
        ready_nodes = self._graph_store.get_ready_nodes()
        if ready_nodes:
            # Resort based on new priorities
            self._scheduler.peek_all(ready_nodes)
            # Priority order is now updated automatically

        return True

    def _handle_external_signal(self, replan_event: ReplanEvent) -> bool:
        """Handle external signal replanning."""
        signal_data = replan_event.metadata.get("signal_data", {})

        # Handle different types of external signals
        if "new_goal" in signal_data:
            # Add new goal to the plan
            new_goal_data = signal_data["new_goal"]
            new_goal = GoalNode(**new_goal_data)
            self._graph_store.add_node(new_goal)

        if "cancel_nodes" in signal_data:
            # Cancel specified nodes
            for node_id in signal_data["cancel_nodes"]:
                if self._graph_store.has_node(node_id):
                    node = self._graph_store.get_node(node_id)
                    node.mark_cancelled()
                    self._graph_store.update_node(node)

        if "reprioritize" in signal_data:
            # Update priorities based on signal
            priority_updates = signal_data["reprioritize"]
            for node_id, new_priority in priority_updates.items():
                if self._graph_store.has_node(node_id):
                    node = self._graph_store.get_node(node_id)
                    node.priority = float(new_priority)
                    self._graph_store.update_node(node)

        return True

    def _handle_generic_replan(self, replan_event: ReplanEvent) -> bool:
        """Handle generic replanning operations."""
        # Default strategy: re-evaluate ready nodes and priorities
        ready_nodes = self._graph_store.get_ready_nodes()

        if ready_nodes:
            # Update priorities based on current context
            for node in ready_nodes:
                # Simple heuristic: boost priority if waiting too long
                wait_time = datetime.now(timezone.utc) - node.created_at
                if wait_time.total_seconds() > 3600:  # More than 1 hour
                    node.priority += 1.0
                    self._graph_store.update_node(node)

        return True

    def get_triggers(self) -> List[ReplanTrigger]:
        """
        Get all registered triggers.

        Returns:
            List of registered triggers
        """
        return self._triggers.copy()

    def get_replanning_stats(self) -> Dict[str, Any]:
        """
        Get replanning statistics.

        Returns:
            Replanning statistics
        """
        return self._replanning_stats.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get replanning statistics (alias for get_replanning_stats).

        Returns:
            Replanning statistics
        """
        stats = self._replanning_stats.copy()
        stats["triggers_registered"] = len(self._triggers)
        return stats
