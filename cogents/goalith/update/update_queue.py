import asyncio
import threading
from queue import Empty, Queue
from typing import List, Optional

from ..base.update_event import UpdateEvent


class UpdateQueue:
    """
    Thread-safe queue for update events.

    Supports both synchronous and asynchronous access patterns.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize update queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue = Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._async_queue: Optional[asyncio.Queue] = None
        self._lock = threading.Lock()
        self._event_count = 0

    @property
    def maxsize(self) -> int:
        """Get the maximum queue size."""
        return self._maxsize

    def put(self, event: UpdateEvent, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Put an update event in the queue.

        Args:
            event: The update event
            block: Whether to block if queue is full
            timeout: Timeout for blocking operations
        """
        with self._lock:
            self._queue.put(event, block=block, timeout=timeout)
            self._event_count += 1

            # Also put in async queue if it exists
            if self._async_queue:
                try:
                    self._async_queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # Ignore async queue overflow

    def get(self, block: bool = True, timeout: Optional[float] = None) -> UpdateEvent:
        """
        Get an update event from the queue.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout for blocking operations

        Returns:
            The next update event

        Raises:
            Empty: If queue is empty and block=False
        """
        return self._queue.get(block=block, timeout=timeout)

    def get_all(self) -> List[UpdateEvent]:
        """
        Get all pending events without blocking.

        Returns:
            List of all pending events
        """
        events = []
        while True:
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events

    def size(self) -> int:
        """
        Get current queue size.

        Returns:
            Number of pending events
        """
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """
        Check if queue is empty.

        Returns:
            True if empty, False otherwise
        """
        return self._queue.empty()

    def get_event_count(self) -> int:
        """
        Get total number of events processed.

        Returns:
            Total event count
        """
        with self._lock:
            return self._event_count

    # Aliases expected by tests
    def qsize(self) -> int:
        """Alias for size()."""
        return self.size()

    def empty(self) -> bool:
        """Alias for is_empty()."""
        return self.is_empty()

    def peek(self) -> Optional[UpdateEvent]:
        """Peek at the next event without removing it."""
        try:
            # Get the event and immediately put it back
            event = self._queue.get_nowait()
            # Put it back at the front (though Queue doesn't support this directly)
            # For simplicity, we'll use a temporary storage
            temp_events = [event]
            while True:
                try:
                    temp_events.append(self._queue.get_nowait())
                except Empty:
                    break
            # Put everything back in order
            for e in temp_events:
                self._queue.put_nowait(e)
            return event
        except Empty:
            return None

    async def get_async(self) -> UpdateEvent:
        """
        Get an event asynchronously.

        Returns:
            The next update event
        """
        if not self._async_queue:
            self._async_queue = asyncio.Queue()

        return await self._async_queue.get()

    def clear(self) -> None:
        """
        Clear all events from the queue.
        """
        with self._lock:
            # Clear the synchronous queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

            # Clear the async queue if it exists
            if self._async_queue:
                while not self._async_queue.empty():
                    try:
                        self._async_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
