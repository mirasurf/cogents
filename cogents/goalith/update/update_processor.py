"""
Update and notification module for the GoalithService.

Handles external updates and broadcasts changes to subscribers.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from ..base.update_event import UpdateEvent, UpdateType


class UpdateProcessor:
    """
    Processes update events and applies them to the graph store.

    Hooks into conflict management before final commit.
    """

    def __init__(self, graph_store, conflict_orchestrator=None):
        """
        Initialize update processor.

        Args:
            graph_store: The graph store to update
            conflict_orchestrator: Optional conflict management
        """
        self._graph_store = graph_store
        self._conflict_orchestrator = conflict_orchestrator
        self._processing_stats = {"total_processed": 0, "by_type": {}, "errors": 0, "conflicts_detected": 0}

    @property
    def graph_store(self):
        """Get the graph store."""
        return self._graph_store

    @property
    def conflict_orchestrator(self):
        """Get the conflict orchestrator."""
        return self._conflict_orchestrator

    def process_event(self, event: UpdateEvent) -> bool:
        """
        Process a single update event.

        Args:
            event: The update event to process

        Returns:
            True if successfully processed, False otherwise
        """
        try:
            # Check for conflicts if orchestrator is available
            if self._conflict_orchestrator:
                conflicts = self._conflict_orchestrator.detect_conflicts([event])
                if conflicts:
                    self._processing_stats["conflicts_detected"] += len(conflicts)
                    resolved = self._conflict_orchestrator.resolve_conflicts(conflicts)
                    if not resolved:
                        return False

            # Apply the update
            success = self._apply_update(event)

            if success:
                self._processing_stats["total_processed"] += 1
                event_type = str(event.update_type)
                self._processing_stats["by_type"][event_type] = self._processing_stats["by_type"].get(event_type, 0) + 1
            else:
                self._processing_stats["errors"] += 1

            return success

        except Exception as e:
            print(f"Error processing update event {event.id}: {e}")
            self._processing_stats["errors"] += 1
            return False

    def _apply_update(self, event: UpdateEvent) -> bool:
        """
        Apply the update to the graph store.

        Args:
            event: The update event

        Returns:
            True if successful, False otherwise
        """
        try:
            if event.update_type == UpdateType.STATUS_CHANGE:
                return self._apply_status_change(event)
            elif event.update_type == UpdateType.NODE_EDIT:
                return self._apply_node_edit(event)
            elif event.update_type == UpdateType.NODE_ADD:
                return self._apply_node_add(event)
            elif event.update_type == UpdateType.NODE_REMOVE:
                return self._apply_node_remove(event)
            elif event.update_type == UpdateType.DEPENDENCY_ADD:
                return self._apply_dependency_add(event)
            elif event.update_type == UpdateType.DEPENDENCY_REMOVE:
                return self._apply_dependency_remove(event)
            elif event.update_type == UpdateType.CONTEXT_UPDATE:
                return self._apply_context_update(event)
            elif event.update_type == UpdateType.PRIORITY_CHANGE:
                return self._apply_priority_change(event)
            else:
                print(f"Unknown update type: {event.update_type}")
                return False

        except Exception as e:
            print(f"Error applying update: {e}")
            return False

    def _apply_status_change(self, event: UpdateEvent) -> bool:
        """Apply status change update."""
        from ..base.goal_node import NodeStatus

        if not self._graph_store.has_node(event.node_id):
            return False

        node = self._graph_store.get_node(event.node_id)
        new_status_value = event.data.get("new_status") if event.data else None
        if new_status_value is None:
            return False
        new_status = NodeStatus(new_status_value)

        if new_status == NodeStatus.IN_PROGRESS:
            node.mark_started()
        elif new_status == NodeStatus.COMPLETED:
            node.mark_completed()
        elif new_status == NodeStatus.FAILED:
            error_msg = event.data.get("error_message")
            node.mark_failed(error_msg)
        elif new_status == NodeStatus.CANCELLED:
            node.mark_cancelled()
        else:
            node.status = new_status
            node.updated_at = datetime.now(timezone.utc)

        self._graph_store.update_node(node)
        return True

    def _apply_node_edit(self, event: UpdateEvent) -> bool:
        """Apply node edit update."""
        if not self._graph_store.has_node(event.node_id):
            return False

        node = self._graph_store.get_node(event.node_id)

        # Update specified fields
        for field, value in event.data.items():
            if hasattr(node, field):
                setattr(node, field, value)

        node.updated_at = datetime.now(timezone.utc)
        self._graph_store.update_node(node)
        return True

    def _apply_node_add(self, event: UpdateEvent) -> bool:
        """Apply node addition update."""
        from ..base.goal_node import GoalNode

        node_data = event.data.get("node")
        if not node_data:
            return False

        # Convert sets back from lists if needed
        if "dependencies" in node_data:
            node_data["dependencies"] = set(node_data["dependencies"])
        if "children" in node_data:
            node_data["children"] = set(node_data["children"])
        if "tags" in node_data:
            node_data["tags"] = set(node_data["tags"])

        node = GoalNode(**node_data)
        self._graph_store.add_node(node)
        return True

    def _apply_node_remove(self, event: UpdateEvent) -> bool:
        """Apply node removal update."""
        if not self._graph_store.has_node(event.node_id):
            return False

        self._graph_store.remove_node(event.node_id)
        return True

    def _apply_dependency_add(self, event: UpdateEvent) -> bool:
        """Apply dependency addition update."""
        # Support both "dependency_id" and "child_id" for backward compatibility
        dependency_id = event.data.get("dependency_id") or event.data.get("child_id")
        if not dependency_id:
            return False

        self._graph_store.add_dependency(event.node_id, dependency_id)
        return True

    def _apply_dependency_remove(self, event: UpdateEvent) -> bool:
        """Apply dependency removal update."""
        # Support both "dependency_id" and "child_id" for backward compatibility
        dependency_id = event.data.get("dependency_id") or event.data.get("child_id")
        if not dependency_id:
            return False

        self._graph_store.remove_dependency(event.node_id, dependency_id)
        return True

    def _apply_context_update(self, event: UpdateEvent) -> bool:
        """Apply context update."""
        if not self._graph_store.has_node(event.node_id):
            return False

        node = self._graph_store.get_node(event.node_id)
        context_updates = event.data.get("context", {})

        for key, value in context_updates.items():
            node.update_context(key, value)

        self._graph_store.update_node(node)
        return True

    def _apply_priority_change(self, event: UpdateEvent) -> bool:
        """Apply priority change update."""
        if not self._graph_store.has_node(event.node_id):
            return False

        node = self._graph_store.get_node(event.node_id)
        new_priority = event.data.get("new_priority")

        if new_priority is not None:
            node.priority = float(new_priority)
            node.updated_at = datetime.now(timezone.utc)
            self._graph_store.update_node(node)
            return True

        return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Processing statistics
        """
        return self._processing_stats.copy()

    # Aliases expected by tests
    def process_update(self, event: UpdateEvent) -> bool:
        """Alias for process_event."""
        return self.process_event(event)

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_processing_stats with test-compatible format."""
        stats = self.get_processing_stats()
        # Add aliases for test compatibility
        stats["updates_processed"] = stats["total_processed"]
        return stats
