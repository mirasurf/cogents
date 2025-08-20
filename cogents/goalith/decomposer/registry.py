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

    def register(self, name: str, decomposer: GoalDecomposer, replace: bool = False) -> None:
        """
        Register a decomposer.

        Args:
            name: Name to register the decomposer under
            decomposer: The decomposer to register
            replace: If True, replace existing decomposer with same name

        Raises:
            ValueError: If decomposer name already exists and replace=False
            TypeError: If decomposer is not a valid GoalDecomposer instance
        """
        if not isinstance(decomposer, GoalDecomposer):
            raise TypeError(f"Decomposer must be an instance of GoalDecomposer, got {type(decomposer)}")

        if name in self._decomposers and not replace:
            raise ValueError(f"Decomposer '{name}' already registered")

        self._decomposers[name] = decomposer

    def register_or_replace(self, name: str, decomposer: GoalDecomposer) -> None:
        """
        Register a decomposer, replacing any existing one with the same name.

        Args:
            name: Name to register the decomposer under
            decomposer: The decomposer to register
        """
        self._decomposers[name] = decomposer

    def register_bulk(self, decomposers: Dict[str, GoalDecomposer]) -> None:
        """
        Register multiple decomposers at once.

        Args:
            decomposers: Dictionary mapping names to decomposer instances
        """
        for name, decomposer in decomposers.items():
            self._decomposers[name] = decomposer

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

    def get_decomposer_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a decomposer.

        Args:
            name: Name of the decomposer

        Returns:
            Dictionary with decomposer information

        Raises:
            ValueError: If decomposer not found
        """
        decomposer = self.get_decomposer(name)
        return {
            "name": name,
            "description": getattr(decomposer, "description", "No description available"),
            "type": type(decomposer).__name__,
            "registered_at": "now",  # Could be enhanced with actual timestamp
        }

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

    def temporary_decomposer(self, name: str, decomposer: GoalDecomposer):
        """
        Context manager for temporary decomposer registration.

        Args:
            name: Name to register the decomposer under
            decomposer: The decomposer to register temporarily
        """

        class TemporaryDecomposerContext:
            def __init__(self, registry, name, decomposer):
                self.registry = registry
                self.name = name
                self.decomposer = decomposer
                self.was_registered = name in registry._decomposers
                self.original_decomposer = registry._decomposers.get(name)

            def __enter__(self):
                self.registry._decomposers[self.name] = self.decomposer
                return self.registry

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.was_registered and self.original_decomposer:
                    self.registry._decomposers[self.name] = self.original_decomposer
                else:
                    self.registry._decomposers.pop(self.name, None)

        return TemporaryDecomposerContext(self, name, decomposer)
