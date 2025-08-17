from typing import Any, Dict, List

from ..base.conflict import Conflict, ConflictSeverity, ConflictType
from ..base.goal_node import NodeStatus
from ..base.update_event import UpdateEvent, UpdateType


class ConflictDetector:
    """
    Detects conflicts in update events and graph states.

    Watches for illegal states, cycles, and semantic inconsistencies.
    """

    def __init__(self, graph_store):
        """
        Initialize conflict detector.

        Args:
            graph_store: The graph store to monitor
        """
        self._graph_store = graph_store
        self._detection_stats = {
            "total_checked": 0,
            "conflicts_found": 0,
            "by_type": {},
        }

    @property
    def graph_store(self):
        """Get the graph store."""
        return self._graph_store

    def detect_conflicts(self, events) -> List[Conflict]:
        """
        Detect conflicts in update events.

        Args:
            events: Update event or list of update events to check

        Returns:
            List of detected conflicts
        """
        # Handle both single event and list of events
        if not isinstance(events, list):
            events = [events]

        self._detection_stats["total_checked"] += len(events)
        conflicts = []

        # Check different types of conflicts
        conflicts.extend(self._check_cycle_conflicts(events))
        conflicts.extend(self._check_concurrent_conflicts(events))
        conflicts.extend(self._check_status_conflicts(events))
        conflicts.extend(self._check_dependency_conflicts(events))
        conflicts.extend(self._check_priority_conflicts(events))

        # Update stats
        self._detection_stats["conflicts_found"] += len(conflicts)
        for conflict in conflicts:
            conflict_type = str(conflict.conflict_type)
            self._detection_stats["by_type"][conflict_type] = self._detection_stats["by_type"].get(conflict_type, 0) + 1

        return conflicts

    def _check_cycle_conflicts(self, events: List[UpdateEvent]) -> List[Conflict]:
        """Check for dependency cycles."""
        conflicts = []

        dependency_adds = [event for event in events if event.update_type == UpdateType.DEPENDENCY_ADD]

        for event in dependency_adds:
            dependent_id = event.node_id
            # Check for both dependency_id and dependency_node_id for compatibility
            dependency_id = event.data.get("dependency_id") or event.data.get("dependency_node_id")

            if not dependency_id:
                continue

            # Check if this would create a cycle
            if self._would_create_cycle(dependent_id, dependency_id):
                conflict = Conflict(
                    conflict_type=ConflictType.CYCLE_DETECTED,
                    severity=ConflictSeverity.CRITICAL,
                    affected_nodes=[dependent_id, dependency_id],
                    description=f"Adding dependency {dependency_id} -> {dependent_id} would create a cycle",
                    events=[event],
                )
                conflicts.append(conflict)

        return conflicts

    def _check_concurrent_conflicts(self, events: List[UpdateEvent]) -> List[Conflict]:
        """Check for concurrent updates to the same node."""
        conflicts = []

        # Group events by node ID
        node_events: Dict[str, List[UpdateEvent]] = {}
        for event in events:
            if event.node_id not in node_events:
                node_events[event.node_id] = []
            node_events[event.node_id].append(event)

        # Check for conflicts in each node's events
        for node_id, node_event_list in node_events.items():
            if len(node_event_list) > 1:
                # Check if these are conflicting updates
                if self._are_conflicting_updates(node_event_list):
                    conflict = Conflict(
                        conflict_type=ConflictType.CONCURRENT_UPDATE,
                        severity=ConflictSeverity.MEDIUM,
                        affected_nodes=[node_id],
                        description=f"Concurrent conflicting updates to node {node_id}",
                        events=node_event_list,
                    )
                    conflicts.append(conflict)

        return conflicts

    def _check_status_conflicts(self, events: List[UpdateEvent]) -> List[Conflict]:
        """Check for status inconsistencies."""
        conflicts = []

        status_changes = [event for event in events if event.update_type == UpdateType.STATUS_CHANGE]

        for event in status_changes:
            node_id = event.node_id
            new_status = event.data.get("new_status")

            if not new_status:
                continue

            try:
                if not self._graph_store.has_node(node_id):
                    continue
                node = self._graph_store.get_node(node_id)
            except Exception:
                # Handle errors gracefully - skip this event if we can't access the node
                continue

            # Check for invalid status transitions
            if not self._is_valid_status_transition(node.status, NodeStatus(new_status)):
                conflict = Conflict(
                    conflict_type=ConflictType.STATUS_INCONSISTENCY,
                    severity=ConflictSeverity.HIGH,
                    affected_nodes=[node_id],
                    description=f"Invalid status transition from {node.status} to {new_status}",
                    events=[event],
                )
                conflicts.append(conflict)

        return conflicts

    def _check_dependency_conflicts(self, events: List[UpdateEvent]) -> List[Conflict]:
        """Check for dependency violations."""
        conflicts = []

        for event in events:
            if event.update_type == UpdateType.STATUS_CHANGE:
                node_id = event.node_id
                new_status = event.data.get("new_status")

                if new_status == NodeStatus.IN_PROGRESS:
                    # Check if all dependencies are completed
                    if self._graph_store.has_node(node_id):
                        node = self._graph_store.get_node(node_id)
                        incomplete_deps = []

                        for dep_id in node.dependencies:
                            if self._graph_store.has_node(dep_id):
                                dep_node = self._graph_store.get_node(dep_id)
                                if dep_node.status != NodeStatus.COMPLETED:
                                    incomplete_deps.append(dep_id)

                        if incomplete_deps:
                            conflict = Conflict(
                                conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                                severity=ConflictSeverity.HIGH,
                                affected_nodes=[node_id] + incomplete_deps,
                                description=f"Node {node_id} cannot start with incomplete dependencies: {incomplete_deps}",
                                events=[event],
                            )
                            conflicts.append(conflict)

        return conflicts

    def _check_priority_conflicts(self, events: List[UpdateEvent]) -> List[Conflict]:
        """Check for priority conflicts."""
        conflicts = []

        priority_changes = [event for event in events if event.update_type == UpdateType.PRIORITY_CHANGE]

        # Group by potential resource conflicts (placeholder logic)
        for event in priority_changes:
            node_id = event.node_id
            new_priority = event.data.get("new_priority")

            if self._graph_store.has_node(node_id):
                self._graph_store.get_node(node_id)

                # Check for unreasonable priority values
                if new_priority is not None and (new_priority < 0 or new_priority > 100):
                    conflict = Conflict(
                        conflict_type=ConflictType.PRIORITY_CONFLICT,
                        severity=ConflictSeverity.LOW,
                        affected_nodes=[node_id],
                        description=f"Priority value {new_priority} is outside reasonable range (0-100)",
                        events=[event],
                    )
                    conflicts.append(conflict)

        return conflicts

    def _would_create_cycle(self, dependent_id: str, dependency_id: str) -> bool:
        """Check if adding a dependency would create a cycle."""
        try:
            # Adding dependency_id as a dependency of dependent_id would create a cycle if:
            # dependent_id is already an ancestor of dependency_id (transitively)

            # Check if dependent_id is in the transitive ancestors of dependency_id
            if hasattr(self._graph_store, "get_ancestors"):
                return self._is_transitive_ancestor(dependent_id, dependency_id)
            else:
                # Fallback: check if dependency_id has a path to dependent_id
                return self._has_path(dependency_id, dependent_id)

        except Exception:
            return True  # Assume conflict if we can't determine

    def _is_transitive_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor_id is a transitive ancestor of descendant_id."""
        visited = set()
        stack = [descendant_id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == ancestor_id:
                return True

            # Get immediate ancestors and add to stack
            try:
                immediate_ancestors = self._graph_store.get_ancestors(current)
                for anc in immediate_ancestors:
                    if anc not in visited:
                        stack.append(anc)
            except:
                pass

        return False

    def _has_path(self, start_id: str, end_id: str) -> bool:
        """Check if there's a path from start to end node."""
        if not self._graph_store.has_node(start_id) or not self._graph_store.has_node(end_id):
            return False

        if start_id == end_id:
            return True

        visited = set()
        stack = [start_id]

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)

            if current == end_id:
                return True

            # Add children to stack
            if self._graph_store.has_node(current):
                node = self._graph_store.get_node(current)
                for child_id in node.children:
                    if child_id not in visited:
                        stack.append(child_id)

        return False

    def _are_conflicting_updates(self, events: List[UpdateEvent]) -> bool:
        """Check if a list of events are conflicting."""
        # Check for conflicting field updates
        field_updates = {}

        for event in events:
            if event.update_type == UpdateType.NODE_EDIT:
                for field in event.data.keys():
                    if field in field_updates:
                        return True  # Same field updated multiple times
                    field_updates[field] = event

        # Check for status conflicts
        status_updates = [event for event in events if event.update_type == UpdateType.STATUS_CHANGE]

        return len(status_updates) > 1

    def _is_valid_status_transition(self, current: NodeStatus, new: NodeStatus) -> bool:
        """Check if a status transition is valid."""
        # Define valid transitions
        valid_transitions = {
            NodeStatus.PENDING: {NodeStatus.IN_PROGRESS, NodeStatus.CANCELLED},
            NodeStatus.IN_PROGRESS: {
                NodeStatus.COMPLETED,
                NodeStatus.FAILED,
                NodeStatus.CANCELLED,
            },
            NodeStatus.COMPLETED: {NodeStatus.PENDING},  # Allow restarting completed tasks
            NodeStatus.FAILED: {
                NodeStatus.PENDING,
                NodeStatus.IN_PROGRESS,
            },  # Allow retry
            NodeStatus.CANCELLED: {NodeStatus.PENDING},  # Allow reactivation
            NodeStatus.BLOCKED: {NodeStatus.PENDING, NodeStatus.CANCELLED},
        }

        return new in valid_transitions.get(current, set())

    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get conflict detection statistics.

        Returns:
            Detection statistics
        """
        return self._detection_stats.copy()
