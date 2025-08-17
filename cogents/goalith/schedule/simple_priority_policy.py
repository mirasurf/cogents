from ..base.goal_node import GoalNode
from ..base.priority_policy import PriorityOrder, PriorityPolicy


class SimplePriorityPolicy(PriorityPolicy):
    """
    Simple priority policy based on node priority values.
    """

    def __init__(
        self,
        order: PriorityOrder = PriorityOrder.HIGHEST_FIRST,
        name: str = "simple_priority",
    ):
        """
        Initialize policy.

        Args:
            order: Priority ordering (highest or lowest first)
            name: Name of this policy
        """
        self._order = order
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this policy."""
        return self._name

    @property
    def order(self) -> PriorityOrder:
        """Get the priority order."""
        return self._order

    def score(self, node: GoalNode) -> float:
        """
        Calculate priority score based on node priority.

        Args:
            node: The node to score

        Returns:
            Priority score
        """
        if self._order == PriorityOrder.HIGHEST_FIRST:
            return node.priority
        else:
            return -node.priority

    def compare(self, node1: GoalNode, node2: GoalNode) -> int:
        """
        Compare nodes based on priority.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Comparison result
        """
        score1 = self.score(node1)
        score2 = self.score(node2)

        if score1 > score2:
            return -1
        elif score1 < score2:
            return 1
        else:
            return 0
