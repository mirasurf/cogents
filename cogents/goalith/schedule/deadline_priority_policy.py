from datetime import datetime, timezone

from ..base.goal_node import GoalNode
from ..base.priority_policy import PriorityPolicy


class DeadlinePriorityPolicy(PriorityPolicy):
    """
    Priority policy that considers deadlines.
    """

    def __init__(self, deadline_weight: float = 20.0, name: str = "deadline_priority"):
        """
        Initialize policy.

        Args:
            deadline_weight: Weight for deadline urgency
            name: Name of this policy
        """
        self._deadline_weight = deadline_weight
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this policy."""
        return self._name

    def score(self, node: GoalNode) -> float:
        """
        Calculate priority score considering deadlines.

        Args:
            node: The node to score

        Returns:
            Priority score
        """
        base_score = node.priority

        if node.deadline:
            now = datetime.now(timezone.utc)
            time_to_deadline = (node.deadline - now).total_seconds()

            # Boost priority for deadlines
            if time_to_deadline > 0:
                # Base deadline boost + urgency boost
                base_deadline_boost = 15.0  # Any deadline gets a base boost
                urgency = 1.0 / (time_to_deadline / 3600 + 1)  # Hours to deadline
                total_boost = base_deadline_boost + (urgency * self._deadline_weight)
                base_score += total_boost
            else:
                # Past deadline - very high priority
                base_score += 50.0

        return base_score

    def compare(self, node1: GoalNode, node2: GoalNode) -> int:
        """
        Compare nodes based on deadline-adjusted priority.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Comparison result
        """
        # Same object comparison
        if node1 is node2:
            return 0

        score1 = self.score(node1)
        score2 = self.score(node2)

        # Use a small tolerance for floating point comparison
        tolerance = 1e-6
        if abs(score1 - score2) <= tolerance:
            return 0
        elif score1 > score2:
            return -1
        else:
            return 1
