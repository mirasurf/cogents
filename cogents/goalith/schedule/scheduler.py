"""
Scheduling and prioritization module for the GoalithService.

Determines which ready nodes to execute based on priority and other constraints.
"""

from typing import Any, Dict, List, Optional

from ..base.goal_node import GoalNode
from ..base.priority_policy import PriorityPolicy
from .simple_priority_policy import SimplePriorityPolicy


class Scheduler:
    """
    Scheduler that applies priority policies to select ready nodes.

    Provides methods to get the next highest-priority node or peek at
    all ready nodes in priority order.
    """

    def __init__(self, priority_policy: Optional[PriorityPolicy] = None, policy: Optional[PriorityPolicy] = None):
        """
        Initialize scheduler.

        Args:
            priority_policy: Policy for prioritization (default: simple priority)
            policy: Alias for priority_policy (for backward compatibility)
        """
        # priority_policy takes precedence over policy
        if priority_policy is not None:
            self._policy = priority_policy
        elif policy is not None:
            self._policy = policy
        else:
            self._policy = SimplePriorityPolicy()

        self._stats = {
            "schedule_calls": 0,
            "nodes_scheduled": 0,
            "get_next_calls": 0,
        }

    def set_policy(self, policy: PriorityPolicy) -> None:
        """
        Set the priority policy.

        Args:
            policy: New priority policy
        """
        self._policy = policy

    def get_policy(self) -> PriorityPolicy:
        """
        Get the current priority policy.

        Returns:
            Current priority policy
        """
        return self._policy

    @property
    def policy(self) -> PriorityPolicy:
        """
        Get the current priority policy (property for backward compatibility).

        Returns:
            Current priority policy
        """
        return self._policy

    def get_next(self, nodes: List[GoalNode]) -> Optional[GoalNode]:
        """
        Get the next highest-priority ready node.

        Args:
            nodes: List of nodes to choose from

        Returns:
            Highest-priority ready node, or None if no ready nodes available
        """
        self._stats["get_next_calls"] += 1

        if not nodes:
            return None

        # Filter for ready nodes (only PENDING and IN_PROGRESS)
        from ..base.goal_node import NodeStatus

        ready_nodes = [node for node in nodes if node.status in [NodeStatus.PENDING, NodeStatus.IN_PROGRESS]]

        if not ready_nodes:
            return None

        if len(ready_nodes) == 1:
            return ready_nodes[0]

        # Find the highest-priority node
        best_node = ready_nodes[0]
        for node in ready_nodes[1:]:
            if self._policy.compare(node, best_node) < 0:
                best_node = node

        self._stats["schedule_calls"] += 1  # get_next is also a scheduling operation
        return best_node

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        for key in self._stats:
            self._stats[key] = 0

    def schedule(self, nodes: List[GoalNode], limit: Optional[int] = None) -> List[GoalNode]:
        """
        Schedule nodes in priority order.

        Args:
            nodes: List of nodes to consider for scheduling
            limit: Maximum number of nodes to return (default: all)

        Returns:
            List of nodes in priority order
        """
        self._stats["schedule_calls"] += 1

        if not nodes:
            return []

        # Use peek_all to get sorted nodes (no filtering by status)
        sorted_nodes = self.peek_all(nodes)

        if limit is not None:
            sorted_nodes = sorted_nodes[:limit]

        self._stats["nodes_scheduled"] += len(sorted_nodes)
        return sorted_nodes

    def peek_all(self, ready_nodes: List[GoalNode]) -> List[GoalNode]:
        """
        Get all ready nodes sorted by priority.

        Args:
            ready_nodes: List of ready nodes to sort

        Returns:
            Nodes sorted by priority (highest first)
        """
        if not ready_nodes:
            return []

        # Sort using the priority policy
        def sort_key(node: GoalNode) -> float:
            return -self._policy.score(node)  # Negative for descending order

        return sorted(ready_nodes, key=sort_key)

    def peek_top_n(self, ready_nodes: List[GoalNode], n: int) -> List[GoalNode]:
        """
        Get the top N highest-priority ready nodes.

        Args:
            ready_nodes: List of ready nodes
            n: Number of top nodes to return

        Returns:
            Top N nodes by priority
        """
        sorted_nodes = self.peek_all(ready_nodes)
        return sorted_nodes[:n]

    def filter_by_criteria(self, ready_nodes: List[GoalNode], criteria: Dict[str, Any]) -> List[GoalNode]:
        """
        Filter ready nodes by additional criteria.

        Args:
            ready_nodes: List of ready nodes
            criteria: Filtering criteria

        Returns:
            Filtered nodes
        """
        filtered = []

        for node in ready_nodes:
            matches = True

            # Check each criterion
            for key, value in criteria.items():
                if key == "type" and node.type != value:
                    matches = False
                    break
                elif key == "assigned_to" and node.assigned_to != value:
                    matches = False
                    break
                elif key == "tags" and not (set(value) & node.tags):
                    matches = False
                    break
                elif key == "min_priority" and node.priority < value:
                    matches = False
                    break
                elif key == "max_priority" and node.priority > value:
                    matches = False
                    break
                elif key.startswith("context."):
                    context_key = key[8:]  # Remove "context." prefix
                    if context_key not in node.context or node.context[context_key] != value:
                        matches = False
                        break

            if matches:
                filtered.append(node)

        return filtered

    def get_next_with_criteria(
        self, ready_nodes: List[GoalNode], criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[GoalNode]:
        """
        Get the next node that matches criteria.

        Args:
            ready_nodes: List of ready nodes
            criteria: Optional filtering criteria

        Returns:
            Highest-priority node matching criteria, or None
        """
        candidates = ready_nodes

        if criteria:
            candidates = self.filter_by_criteria(ready_nodes, criteria)

        return self.get_next(candidates)

    def get_scheduling_stats(self, ready_nodes: List[GoalNode]) -> Dict[str, Any]:
        """
        Get statistics about scheduling state.

        Args:
            ready_nodes: List of ready nodes

        Returns:
            Scheduling statistics
        """
        if not ready_nodes:
            return {
                "total_ready": 0,
                "policy": self._policy.name,
                "top_priority": None,
                "priority_range": None,
            }

        scores = [self._policy.score(node) for node in ready_nodes]
        top_node = self.get_next(ready_nodes)

        return {
            "total_ready": len(ready_nodes),
            "policy": self._policy.name,
            "top_priority": top_node.priority if top_node else None,
            "priority_range": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            },
            "by_type": {
                str(node_type): len([n for n in ready_nodes if n.type == node_type])
                for node_type in {n.type for n in ready_nodes}
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        return self._stats.copy()
