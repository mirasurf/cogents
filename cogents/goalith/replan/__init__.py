"""
Replanning module for the GoalithService.

Reactively adjusts plans when things go awry through triggers and replanning strategies.
"""
from datetime import timedelta
from typing import List

from ..base.replanning import ReplanTrigger
from .deadline_trigger import DeadlineTrigger
from .external_signal_trigger import ExternalSignalTrigger
from .priority_change_trigger import PriorityChangeTrigger
from .replanner import Replanner
from .task_fail_trigger import TaskFailureTrigger


# Default set of triggers
def create_default_triggers() -> List[ReplanTrigger]:
    """
    Create a default set of replan triggers.

    Returns:
        List of default triggers
    """
    return [
        TaskFailureTrigger(max_retries=3),
        DeadlineTrigger(warning_threshold=timedelta(hours=1)),
        PriorityChangeTrigger(threshold=2.0),
        ExternalSignalTrigger(),
    ]


__all__ = ["create_default_triggers", "Replanner"]
