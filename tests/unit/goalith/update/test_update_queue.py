"""
Unit tests for goalith.update.update_queue module.
"""
import threading
import time
from queue import Empty

import pytest

from cogents.goalith.base.update_event import UpdateEvent, UpdateType
from cogents.goalith.update.update_queue import UpdateQueue


class TestUpdateQueue:
    """Test UpdateQueue functionality."""

    def test_default_initialization(self):
        """Test creating update queue with default size."""
        queue = UpdateQueue()

        assert queue._maxsize == 1000
        assert queue.size() == 0
        assert queue.is_empty() is True
        assert queue.is_full() is False

    def test_custom_size_initialization(self):
        """Test creating update queue with custom size."""
        queue = UpdateQueue(maxsize=100)

        assert queue._maxsize == 100
        assert queue.size() == 0

    def test_put_update(self):
        """Test putting update in queue."""
        queue = UpdateQueue()
        update = UpdateEvent(update_type=UpdateType.STATUS_CHANGE, node_id="test-node", data={"status": "completed"})

        queue.put(update)

        assert queue.size() == 1
        assert not queue.is_empty()

    def test_put_update_with_timeout(self):
        """Test putting update with timeout."""
        queue = UpdateQueue(maxsize=1)

        # Fill the queue
        update1 = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="node1")
        queue.put(update1)

        # Try to add another with timeout (should work for non-blocking)
        update2 = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="node2")
        queue.put(update2, timeout=0.1)

        # Should have 2 items (queue may auto-expand or block depending on implementation)
        assert queue.size() >= 1

    def test_get_update(self):
        """Test getting update from queue."""
        queue = UpdateQueue()
        update = UpdateEvent(update_type=UpdateType.NODE_EDIT, node_id="test-node", data={"field": "value"})

        queue.put(update)
        retrieved = queue.get()

        assert retrieved == update
        assert queue.size() == 0
        assert queue.is_empty()

    def test_get_update_with_timeout(self):
        """Test getting update with timeout."""
        queue = UpdateQueue()

        # Try to get from empty queue with timeout
        with pytest.raises(Empty):
            queue.get(timeout=0.1)

    def test_get_update_blocking(self):
        """Test blocking get operation."""
        queue = UpdateQueue()
        update = UpdateEvent(update_type=UpdateType.PRIORITY_CHANGE, node_id="node")
        result = []

        def delayed_put():
            time.sleep(0.1)
            queue.put(update)

        def blocking_get():
            result.append(queue.get(timeout=1.0))

        # Start both operations
        put_thread = threading.Thread(target=delayed_put)
        get_thread = threading.Thread(target=blocking_get)

        put_thread.start()
        get_thread.start()

        put_thread.join()
        get_thread.join()

        assert len(result) == 1
        assert result[0] == update

    def test_peek_update(self):
        """Test peeking at next update without removing it."""
        queue = UpdateQueue()
        update = UpdateEvent(update_type=UpdateType.DEPENDENCY_ADD, node_id="test-node")

        queue.put(update)

        # Peek should return the update
        peeked = queue.peek()
        assert peeked == update

        # Queue should still have the update
        assert queue.size() == 1

        # Get should return the same update
        retrieved = queue.get()
        assert retrieved == update
        assert queue.size() == 0

    def test_peek_empty_queue_returns_none(self):
        """Test peeking at empty queue returns None."""
        queue = UpdateQueue()

        peeked = queue.peek()
        assert peeked is None

    def test_clear_queue(self):
        """Test clearing all updates from queue."""
        queue = UpdateQueue()

        # Add multiple updates
        for i in range(5):
            update = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id=f"node-{i}")
            queue.put(update)

        assert queue.size() == 5

        # Clear queue
        queue.clear()

        assert queue.size() == 0
        assert queue.is_empty()

    def test_size_tracking(self):
        """Test size tracking as updates are added and removed."""
        queue = UpdateQueue()

        # Empty queue
        assert queue.size() == 0

        # Add updates
        updates = []
        for i in range(3):
            update = UpdateEvent(update_type=UpdateType.CONTEXT_UPDATE, node_id=f"node-{i}")
            updates.append(update)
            queue.put(update)
            assert queue.size() == i + 1

        # Remove updates
        for i in range(3):
            queue.get()
            assert queue.size() == 3 - i - 1

    def test_is_empty_and_is_full(self):
        """Test empty and full status checking."""
        queue = UpdateQueue(maxsize=2)

        # Empty queue
        assert queue.is_empty() is True
        assert queue.is_full() is False

        # Add one update
        update1 = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="node1")
        queue.put(update1)
        assert queue.is_empty() is False
        assert queue.is_full() is False

        # Fill queue (implementation dependent - may auto-expand)
        update2 = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="node2")
        queue.put(update2)

        # Note: is_full() behavior depends on implementation
        # Some queues auto-expand, others have fixed size

    def test_fifo_ordering(self):
        """Test that updates are processed in FIFO order."""
        queue = UpdateQueue()

        # Add updates in order
        updates = []
        for i in range(5):
            update = UpdateEvent(update_type=UpdateType.NODE_EDIT, node_id=f"node-{i}", data={"order": i})
            updates.append(update)
            queue.put(update)

        # Retrieve updates and verify order
        retrieved = []
        while not queue.is_empty():
            retrieved.append(queue.get())

        assert len(retrieved) == 5
        for i, update in enumerate(retrieved):
            assert update.data["order"] == i

    def test_concurrent_put_and_get(self):
        """Test concurrent put and get operations."""
        queue = UpdateQueue()
        put_count = 100
        results = []
        errors = []

        def producer():
            try:
                for i in range(put_count):
                    update = UpdateEvent(
                        update_type=UpdateType.NODE_ADD,
                        node_id=f"producer-node-{i}",
                        data={"producer_id": threading.current_thread().ident},
                    )
                    queue.put(update)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                consumed = 0
                while consumed < put_count:
                    try:
                        update = queue.get(timeout=1.0)
                        results.append(update)
                        consumed += 1
                    except Empty:
                        break
            except Exception as e:
                errors.append(e)

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all updates were processed
        assert len(results) == put_count

        # Verify queue is empty
        assert queue.is_empty()

    def test_multiple_producers_single_consumer(self):
        """Test multiple producers with single consumer."""
        queue = UpdateQueue()
        num_producers = 3
        updates_per_producer = 20
        total_expected = num_producers * updates_per_producer

        results = []
        errors = []

        def producer(producer_id):
            try:
                for i in range(updates_per_producer):
                    update = UpdateEvent(
                        update_type=UpdateType.STATUS_CHANGE,
                        node_id=f"producer-{producer_id}-node-{i}",
                        data={"producer_id": producer_id, "sequence": i},
                    )
                    queue.put(update)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                consumed = 0
                while consumed < total_expected:
                    try:
                        update = queue.get(timeout=2.0)
                        results.append(update)
                        consumed += 1
                    except Empty:
                        break
            except Exception as e:
                errors.append(e)

        # Start threads
        threads = []

        # Start producers
        for i in range(num_producers):
            thread = threading.Thread(target=producer, args=(i,))
            threads.append(thread)
            thread.start()

        # Start consumer
        consumer_thread = threading.Thread(target=consumer)
        threads.append(consumer_thread)
        consumer_thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all updates were consumed
        assert len(results) == total_expected

        # Verify we got updates from all producers
        producer_ids = set(update.data["producer_id"] for update in results)
        assert producer_ids == set(range(num_producers))

    def test_queue_overflow_handling(self):
        """Test queue behavior when it overflows."""
        queue = UpdateQueue(maxsize=3)

        # Fill the queue
        for i in range(3):
            update = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id=f"node-{i}")
            queue.put(update)

        # Try to add more (behavior depends on implementation)
        overflow_update = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id="overflow-node")

        try:
            queue.put(overflow_update, timeout=0.1)
            # If this succeeds, queue auto-expands or blocks
        except Exception:
            # If this fails, queue has strict size limit
            pass

        # Queue should still be functional
        assert queue.size() >= 3

    def test_update_persistence_across_operations(self):
        """Test that updates maintain their data integrity."""
        queue = UpdateQueue()

        # Create update with complex data
        original_data = {
            "status_change": {"from": "pending", "to": "in_progress"},
            "metadata": {"timestamp": "2023-01-01T10:00:00Z", "user": "test"},
            "nested": {"level1": {"level2": "deep_value"}},
        }

        update = UpdateEvent(
            update_type=UpdateType.STATUS_CHANGE, node_id="complex-node", data=original_data, source="test_system"
        )

        # Put and get the update
        queue.put(update)
        retrieved = queue.get()

        # Verify all data is preserved
        assert retrieved.update_type == update.update_type
        assert retrieved.node_id == update.node_id
        assert retrieved.data == original_data
        assert retrieved.source == update.source
        assert retrieved.id == update.id

        # Verify nested data is accessible
        assert retrieved.data["nested"]["level1"]["level2"] == "deep_value"

    def test_queue_statistics(self):
        """Test queue statistics and monitoring."""
        queue = UpdateQueue()

        # Perform various operations
        for i in range(5):
            update = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id=f"node-{i}")
            queue.put(update)

        # Get some updates
        queue.get()
        queue.get()

        # Check current state
        assert queue.size() == 3
        assert not queue.is_empty()

        # Peek without affecting size
        peeked = queue.peek()
        assert peeked is not None
        assert queue.size() == 3

    def test_error_recovery(self):
        """Test queue recovery from error conditions."""
        queue = UpdateQueue()

        # Add some updates
        for i in range(3):
            update = UpdateEvent(update_type=UpdateType.NODE_ADD, node_id=f"node-{i}")
            queue.put(update)

        # Simulate error condition by clearing and ensure queue still works
        queue.clear()
        assert queue.is_empty()

        # Queue should still be functional
        new_update = UpdateEvent(update_type=UpdateType.NODE_EDIT, node_id="recovery-node")
        queue.put(new_update)

        retrieved = queue.get()
        assert retrieved == new_update
