from typing import Any, Dict, Optional

from ..base.conflict import Conflict, ConflictResolver, ConflictType


class AutomaticConflictResolver(ConflictResolver):
    """
    Automatic resolver that handles simple conflicts with predefined rules.
    """

    def __init__(self, name: str = "automatic_resolver"):
        """
        Initialize automatic resolver.

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
        Resolve conflict using automatic rules.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution dictionary with action and details
        """
        resolution_action = None

        if conflict.conflict_type == ConflictType.CYCLE_DETECTED:
            resolution_action = "reject_dependency_add"

        elif conflict.conflict_type == ConflictType.CONCURRENT_UPDATE:
            # Use timestamp-based resolution (last writer wins)
            resolution_action = "use_latest_update"

        elif conflict.conflict_type == ConflictType.STATUS_INCONSISTENCY:
            resolution_action = "reject_status_change"

        elif conflict.conflict_type == ConflictType.DEPENDENCY_VIOLATION:
            resolution_action = "defer_status_change"

        elif conflict.conflict_type == ConflictType.PRIORITY_CONFLICT:
            resolution_action = "clamp_priority_value"

        else:
            # Default fallback for unsupported conflict types
            resolution_action = "defer_for_manual_review"

        return {
            "resolution": resolution_action,
            "resolver": self.name,
            "conflict_type": str(conflict.conflict_type),
            "automatic": True,
        }
