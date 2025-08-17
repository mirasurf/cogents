from typing import Any, Dict, List, Optional

from ..base.decomposer import GoalDecomposer
from ..base.goal_node import GoalNode


class HumanDecomposer(GoalDecomposer):
    """
    Decomposer that prompts for human input.

    This is a placeholder - would need actual UI integration.
    """

    def __init__(self, name: str = "human"):
        """
        Initialize human decomposer.

        Args:
            name: Name of this decomposer
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose using human input.

        Args:
            goal_node: The goal node to decompose
            context: Optional context

        Returns:
            List of subgoal/task nodes

        Raises:
            DecompositionError: If human input fails
        """
        # This is a placeholder - human decomposition requires UI integration
        raise NotImplementedError("Human intervention required for decomposition")
