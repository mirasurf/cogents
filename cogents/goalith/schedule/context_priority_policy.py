from typing import Any, Dict, Optional

from ..base.goal_node import GoalNode
from ..base.priority_policy import PriorityPolicy


class ContextualPriorityPolicy(PriorityPolicy):
    """
    Priority policy that considers context factors.
    """

    def __init__(
        self,
        context_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        name: str = "contextual_priority",
    ):
        """
        Initialize policy.

        Args:
            context_weights: Weights for different context factors
            context: Contextual configuration for prioritization
            name: Name of this policy
        """
        self._context_weights = context_weights or {
            "importance": 1.0,
            "urgency": 1.5,
            "complexity": -0.5,  # Prefer simpler tasks
            "dependencies": -0.2,  # Prefer tasks with fewer dependencies
        }
        self._context = context or {}
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this policy."""
        return self._name

    def score(self, node: GoalNode) -> float:
        """
        Calculate priority score considering context factors.

        Args:
            node: The node to score

        Returns:
            Priority score
        """
        score = node.priority

        # Apply context weights
        for factor, weight in self._context_weights.items():
            if factor in node.context:
                factor_value = node.context[factor]
                if isinstance(factor_value, (int, float)):
                    score += factor_value * weight

        # Consider dependency count
        if "dependencies" in self._context_weights:
            dep_penalty = len(node.dependencies) * self._context_weights["dependencies"]
            score += dep_penalty

        # Apply context-based boosts
        if "focus_tags" in self._context:
            focus_tags = self._context["focus_tags"]
            if any(tag in node.tags for tag in focus_tags):
                score += 5.0  # Boost for matching focus tags

        if "boost_goals" in self._context and self._context["boost_goals"]:
            from ..base.goal_node import NodeType

            if node.type == NodeType.GOAL:
                score += 3.0  # Boost for goal-type nodes

        return score

    def compare(self, node1: GoalNode, node2: GoalNode) -> int:
        """
        Compare nodes based on contextual priority.

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
