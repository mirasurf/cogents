"""
Unit tests for goalith.base.notification module.
"""
import pytest

from cogents.goalith.base.notification import NotificationSubscriber
from cogents.goalith.base.update_event import UpdateEvent, UpdateType


class ConcreteSubscriber(NotificationSubscriber):
    """Concrete implementation for testing abstract base class."""

    def __init__(self):
        self.received_events = []

    @property
    def subscriber_id(self) -> str:
        """Get unique subscriber ID."""
        return "concrete_test_subscriber"

    def notify(self, event):
        """Store received events for testing."""
        self.received_events.append(event)


class TestNotificationSubscriber:
    """Test NotificationSubscriber abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that NotificationSubscriber cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NotificationSubscriber()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works correctly."""
        subscriber = ConcreteSubscriber()

        event = UpdateEvent(update_type=UpdateType.STATUS_CHANGE, node_id="test-node", data={"status": "completed"})

        subscriber.notify(event)

        assert len(subscriber.received_events) == 1
        assert subscriber.received_events[0] == event

    def test_multiple_notifications(self):
        """Test handling multiple notifications."""
        subscriber = ConcreteSubscriber()

        events = [
            UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="node1"),
            UpdateEvent(update_type=UpdateType.NODE_EDIT, node_id="node2"),
            UpdateEvent(update_type=UpdateType.NODE_REMOVE, node_id="node3"),
        ]

        for event in events:
            subscriber.notify(event)

        assert len(subscriber.received_events) == 3
        assert subscriber.received_events == events

    def test_notify_method_signature(self):
        """Test that notify method has correct signature."""
        subscriber = ConcreteSubscriber()

        # Should accept any object as event
        subscriber.notify("string_event")
        subscriber.notify({"dict": "event"})
        subscriber.notify(123)

        assert len(subscriber.received_events) == 3
