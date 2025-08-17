from typing import Any, Dict, List, Optional

from ..base.decomposer import GoalDecomposer
from ..base.errors import DecompositionError
from ..base.goal_node import GoalNode


class DecomposerRegistry:
    """
    Registry for managing goal decomposers.

    Allows registration, lookup, and invocation of different decomposer
    implementations by name.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._decomposers: Dict[str, GoalDecomposer] = {}

    def register(self, decomposer: GoalDecomposer, replace: bool = False) -> None:
        """
        Register a decomposer.

        Args:
            decomposer: The decomposer to register
            replace: If True, replace existing decomposer with same name

        Raises:
            ValueError: If decomposer name already exists and replace=False
        """
        if decomposer.name in self._decomposers and not replace:
            raise ValueError(f"Decomposer '{decomposer.name}' already registered")

        self._decomposers[decomposer.name] = decomposer

    def register_or_replace(self, decomposer: GoalDecomposer) -> None:
        """
        Register a decomposer, replacing any existing one with the same name.

        Args:
            decomposer: The decomposer to register
        """
        self._decomposers[decomposer.name] = decomposer

    def unregister(self, name: str) -> None:
        """
        Unregister a decomposer.

        Args:
            name: Name of the decomposer to unregister

        Raises:
            ValueError: If decomposer not found
        """
        if name not in self._decomposers:
            raise ValueError(f"Decomposer '{name}' not found")
        del self._decomposers[name]

    def get_decomposer(self, name: str) -> GoalDecomposer:
        """
        Get a decomposer by name.

        Args:
            name: Name of the decomposer

        Returns:
            The requested decomposer

        Raises:
            ValueError: If decomposer not found
        """
        if name not in self._decomposers:
            raise ValueError(f"Decomposer '{name}' not found")

        return self._decomposers[name]

    def has_decomposer(self, name: str) -> bool:
        """
        Check if a decomposer is registered.

        Args:
            name: Name to check

        Returns:
            True if decomposer exists, False otherwise
        """
        return name in self._decomposers

    def list_decomposers(self) -> List[str]:
        """
        Get names of all registered decomposers.

        Returns:
            List of decomposer names
        """
        return list(self._decomposers.keys())

    def decompose(
        self,
        decomposer_name: str,
        goal_node: GoalNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GoalNode]:
        """
        Decompose a goal using the specified decomposer.

        Args:
            decomposer_name: Name of the decomposer to use
            goal_node: The goal node to decompose
            context: Optional context for decomposition

        Returns:
            List of subgoal/task nodes

        Raises:
            ValueError: If decomposer not found
            DecompositionError: If decomposition fails
        """
        decomposer = self.get_decomposer(decomposer_name)

        try:
            subgoals = decomposer.decompose(goal_node, context)

            # Mark the original goal as having been decomposed
            goal_node.decomposer_name = decomposer_name
            goal_node.update_context("decomposed", True)
            goal_node.update_context("decomposition_timestamp", goal_node.updated_at.isoformat())

            # Set up parent-child relationships
            for subgoal in subgoals:
                subgoal.parent = goal_node.id
                goal_node.add_child(subgoal.id)

            return subgoals

        except NotImplementedError:
            # Let NotImplementedError propagate as-is for test compatibility
            raise
        except Exception as e:
            raise DecompositionError(f"Decomposition with '{decomposer_name}' failed: {e}") from e

    def get_registry_info(self) -> Dict[str, str]:
        """
        Get information about all registered decomposers.

        Returns:
            Dictionary mapping decomposer names to descriptions
        """
        return {name: decomposer.description for name, decomposer in self._decomposers.items()}

    # Convenience methods expected by tests
    def get(self, name: str) -> GoalDecomposer:
        """Alias for get_decomposer."""
        return self.get_decomposer(name)

    def has(self, name: str) -> bool:
        """Alias for has_decomposer."""
        return self.has_decomposer(name)

    def clear(self) -> None:
        """Clear all registered decomposers."""
        self._decomposers.clear()
