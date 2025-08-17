from abc import ABC, abstractmethod

from .update_event import UpdateEvent


class NotificationSubscriber(ABC):
    """
    Abstract interface for notification subscribers.
    """

    @abstractmethod
    def notify(self, event: UpdateEvent) -> None:
        """
        Handle a notification event.

        Args:
            event: The update event
        """

    @property
    @abstractmethod
    def subscriber_id(self) -> str:
        """Get unique subscriber ID."""
