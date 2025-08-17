"""
Conflict management module for the GoalithService.

Detects and resolves conflicts from concurrent updates or semantic inconsistencies.
"""

from typing import Any, Dict, List, Optional

from ..base.conflict import Conflict, ConflictResolver
from ..base.update_event import UpdateEvent
from .auto_resolver import AutomaticConflictResolver
from .detector import ConflictDetector


class ConflictOrchestrator:
    """
    Coordinates conflict detection and resolution.

    Manages the pipeline from detection through resolution to application.
    """

    def __init__(self, detector: ConflictDetector, resolver: Optional[ConflictResolver] = None):
        """
        Initialize conflict orchestrator.

        Args:
            detector: Conflict detector
            resolver: Optional conflict resolver (default: automatic)
        """
        self._detector = detector
        self._resolver = resolver or AutomaticConflictResolver()
        self._resolution_stats = {
            "total_conflicts": 0,
            "resolved": 0,
            "unresolved": 0,
            "by_type": {},
        }

    @property
    def detector(self) -> ConflictDetector:
        """Get the conflict detector."""
        return self._detector

    @property
    def resolver(self) -> ConflictResolver:
        """Get the conflict resolver."""
        return self._resolver

    def detect_conflicts(self, events) -> List[Conflict]:
        """
        Detect conflicts in update events.

        Args:
            events: Update event or list of update events

        Returns:
            List of detected conflicts
        """
        return self._detector.detect_conflicts(events)

    def resolve_conflicts(self, conflicts: List[Conflict]) -> bool:
        """
        Resolve a list of conflicts.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            True if all conflicts resolved, False otherwise
        """
        all_resolved = True

        for conflict in conflicts:
            self._resolution_stats["total_conflicts"] += 1
            conflict_type = str(conflict.conflict_type)
            self._resolution_stats["by_type"][conflict_type] = (
                self._resolution_stats["by_type"].get(conflict_type, 0) + 1
            )

            try:
                resolution_action = self._resolver.resolve(conflict)

                if resolution_action:
                    conflict.resolved = True
                    conflict.resolution_action = resolution_action
                    self._resolution_stats["resolved"] += 1
                else:
                    all_resolved = False
                    self._resolution_stats["unresolved"] += 1
            except Exception as e:
                # Handle resolver errors gracefully
                all_resolved = False
                self._resolution_stats["unresolved"] += 1
                conflict.resolution_action = {"error": str(e), "resolver_failed": True}

        return all_resolved

    def process_with_conflict_check(self, events: List[UpdateEvent]) -> tuple[List[Conflict], bool]:
        """
        Process events with conflict detection and resolution.

        Args:
            events: List of update events

        Returns:
            Tuple of (conflicts found, all resolved)
        """
        conflicts = self.detect_conflicts(events)

        if not conflicts:
            return [], True

        all_resolved = self.resolve_conflicts(conflicts)
        return conflicts, all_resolved

    def set_resolver(self, resolver: ConflictResolver) -> None:
        """
        Set the conflict resolver.

        Args:
            resolver: New conflict resolver
        """
        self._resolver = resolver

    def get_resolver(self) -> ConflictResolver:
        """
        Get the current conflict resolver.

        Returns:
            Current resolver
        """
        return self._resolver

    def get_resolution_stats(self) -> Dict[str, Any]:
        """
        Get conflict resolution statistics.

        Returns:
            Resolution statistics
        """
        return self._resolution_stats.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get conflict resolution statistics (alias for get_resolution_stats).

        Returns:
            Resolution statistics
        """
        stats = self.get_resolution_stats()
        # Add compatibility aliases for tests
        stats["conflicts_detected"] = stats["total_conflicts"]
        stats["conflicts_resolved"] = stats["resolved"]
        return stats

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get combined system statistics.

        Returns:
            Combined statistics from detector and orchestrator
        """
        return {
            "detection": self._detector.get_detection_stats(),
            "resolution": self.get_resolution_stats(),
            "resolver": self._resolver.name,
        }

    def detect_and_resolve(self, events) -> List[Conflict]:
        """
        Detect and resolve conflicts in one operation.

        Args:
            events: Update event or list of update events

        Returns:
            List of conflicts (resolved and unresolved)
        """
        conflicts = self.detect_conflicts(events)
        if conflicts:
            self.resolve_conflicts(conflicts)
        return conflicts
