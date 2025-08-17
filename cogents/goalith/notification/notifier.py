from typing import Any, Dict, List, Optional, Set

from ..base.notification import NotificationSubscriber
from ..base.update_event import UpdateEvent, UpdateType


class Notifier:
    """
    Observer pattern implementation for broadcasting update events.

    Manages subscribers and emits events when nodes change.
    """

    def __init__(self):
        """Initialize notifier."""
        self._subscribers: Dict[str, NotificationSubscriber] = {}
        self._type_filters: Dict[str, Set[UpdateType]] = {}
        self._node_filters: Dict[str, Set[str]] = {}
        self._notification_count = 0

    def subscribe(
        self,
        subscriber: NotificationSubscriber,
        event_types: Optional[Set[UpdateType]] = None,
        node_ids: Optional[Set[str]] = None,
    ) -> None:
        """
        Subscribe to notifications.

        Args:
            subscriber: The subscriber
            event_types: Optional filter for event types
            node_ids: Optional filter for specific nodes
        """
        subscriber_id = subscriber.subscriber_id
        self._subscribers[subscriber_id] = subscriber

        if event_types:
            self._type_filters[subscriber_id] = event_types

        if node_ids:
            self._node_filters[subscriber_id] = node_ids

    def unsubscribe(self, subscriber_id: str) -> None:
        """
        Unsubscribe from notifications.

        Args:
            subscriber_id: ID of subscriber to remove
        """
        self._subscribers.pop(subscriber_id, None)
        self._type_filters.pop(subscriber_id, None)
        self._node_filters.pop(subscriber_id, None)

    def notify(self, event: UpdateEvent) -> None:
        """
        Notify all relevant subscribers of an event.

        Args:
            event: The update event to broadcast
        """
        self._notification_count += 1

        for subscriber_id, subscriber in self._subscribers.items():
            # Check type filter
            if subscriber_id in self._type_filters:
                if event.update_type not in self._type_filters[subscriber_id]:
                    continue

            # Check node filter
            if subscriber_id in self._node_filters:
                if event.node_id not in self._node_filters[subscriber_id]:
                    continue

            # Notify subscriber
            try:
                subscriber.notify(event)
            except Exception as e:
                print(f"Error notifying subscriber {subscriber_id}: {e}")

    def notify_batch(self, events: List[UpdateEvent]) -> None:
        """
        Notify subscribers of multiple events.

        Args:
            events: List of update events
        """
        for event in events:
            self.notify(event)

    def get_subscriber_count(self) -> int:
        """
        Get number of active subscribers.

        Returns:
            Number of subscribers
        """
        return len(self._subscribers)

    def get_notification_count(self) -> int:
        """
        Get total number of notifications sent.

        Returns:
            Total notification count
        """
        return self._notification_count

    def get_subscriber_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about subscribers.

        Returns:
            Dictionary with subscriber information
        """
        info = {}
        for subscriber_id in self._subscribers:
            info[subscriber_id] = {
                "type_filter": list(self._type_filters.get(subscriber_id, [])),
                "node_filter": list(self._node_filters.get(subscriber_id, [])),
            }
        return info

    def get_subscribers(self) -> Dict[str, NotificationSubscriber]:
        """Get all subscribers (alias for dict access)."""
        return self._subscribers.copy()

    def clear_subscribers(self) -> None:
        """
        Clear all subscribers.
        """
        self._subscribers.clear()
        self._type_filters.clear()
        self._node_filters.clear()
