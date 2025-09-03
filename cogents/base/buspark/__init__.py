"""Generic event-watchdog framework based on bubus library."""

from bubus import BaseEvent, EventBus

from .base_watchdog import BaseWatchdog, EventProcessor

__all__ = [
    "BaseEvent",
    "BaseWatchdog",
    "EventBus",
    "EventProcessor",
]
