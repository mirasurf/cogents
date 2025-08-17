from typing import Callable

from ..base.notification import NotificationSubscriber
from ..base.update_event import UpdateEvent


class CallableSubscriber(NotificationSubscriber):
    """
    Subscriber that wraps a callable function.
    """

    def __init__(self, callback: Callable[[UpdateEvent], None], subscriber_id: str):
        """
        Initialize callable subscriber.

        Args:
            callback: Function to call on notifications
            subscriber_id: Unique subscriber ID
        """
        self._callback = callback
        self._subscriber_id = subscriber_id

    def notify(self, event: UpdateEvent) -> None:
        """Call the callback function."""
        try:
            self._callback(event)
        except Exception as e:
            print(f"Error in subscriber {self._subscriber_id}: {e}")

    @property
    def subscriber_id(self) -> str:
        """Get subscriber ID."""
        return self._subscriber_id
