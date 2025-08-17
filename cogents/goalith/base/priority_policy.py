from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from .goal_node import GoalNode


class PriorityOrder(str, Enum):
    """Priority ordering options."""

    HIGHEST_FIRST = "highest_first"
    LOWEST_FIRST = "lowest_first"


class PriorityPolicy(ABC):
    """
    Abstract interface for priority policies.

    Encapsulates rules for comparing and ordering nodes based on
    priority and other factors.
    """

    @abstractmethod
    def compare(self, node1: GoalNode, node2: GoalNode) -> int:
        """
        Compare two nodes for priority ordering.

        Args:
            node1: First node to compare
            node2: Second node to compare

        Returns:
            -1 if node1 has higher priority than node2
             0 if they have equal priority
             1 if node2 has higher priority than node1
        """

    @abstractmethod
    def score(self, node: GoalNode) -> float:
        """
        Calculate a priority score for a node.

        Args:
            node: The node to score

        Returns:
            Priority score (higher = more important)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this policy."""

    def sort_nodes(self, nodes: List[GoalNode]) -> List[GoalNode]:
        """
        Sort nodes by priority using this policy.

        Args:
            nodes: List of nodes to sort

        Returns:
            Sorted list of nodes (highest priority first)
        """
        return sorted(nodes, key=lambda node: self.score(node), reverse=True)
