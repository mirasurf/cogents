from typing import Any, Dict, Optional

from ..base.conflict import Conflict, ConflictResolver


class HumanConflictResolver(ConflictResolver):
    """
    Resolver that prompts for human input.

    This is a placeholder - would need actual UI integration.
    """

    def __init__(self, name: str = "human_resolver"):
        """
        Initialize human resolver.

        Args:
            name: Name of this resolver
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this resolver."""
        return self._name

    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """
        Resolve conflict with human input.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution action description
        """
        # Placeholder implementation - always raises NotImplementedError
        # In real implementation, would present UI and collect input
        raise NotImplementedError("Human intervention required but UI not implemented")
