from typing import Any, Dict, List, Optional

from ..base.decomposer import GoalDecomposer
from ..base.goal_node import GoalNode, NodeType


class SimpleListDecomposer(GoalDecomposer):
    """
    Simple decomposer that takes a list of subtask descriptions.

    Useful for manual decomposition or simple cases.
    """

    def __init__(self, subtasks: List[str], name: str = "simple_list"):
        """
        Initialize with a list of subtask descriptions.

        Args:
            subtasks: List of subtask descriptions
            name: Name of this decomposer instance
        """
        self._subtasks = subtasks
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose goal into the predefined subtasks.

        Args:
            goal_node: The goal node to decompose
            context: Optional context (unused)

        Returns:
            List of task nodes
        """
        nodes = []
        for i, subtask_desc in enumerate(self._subtasks):
            task_node = GoalNode(
                description=subtask_desc,
                type=NodeType.TASK,
                parent=goal_node.id,
                priority=goal_node.priority,
                context=goal_node.context.copy() if goal_node.context else {},
                decomposer_name=self.name,
            )
            nodes.append(task_node)

        return nodes
