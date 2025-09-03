"""Base watchdog class for general event monitoring components."""

import time
from collections.abc import Iterable
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from bubus import BaseEvent, EventBus
from pydantic import BaseModel, ConfigDict, Field

# Generic type for the event processor (replaces BrowserSession)
TEventProcessor = TypeVar("TEventProcessor", bound="EventProcessor")


class EventProcessor(Protocol):
    """Protocol defining the interface for event processors.

    This replaces the BrowserSession dependency with a generic interface
    that any event processor must implement.
    """

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus instance."""
        ...

    @property
    def logger(self):
        """Get the logger instance."""
        ...


class BaseWatchdog(BaseModel, Generic[TEventProcessor]):
    """Base class for all event watchdogs.

    Watchdogs monitor events and emit new events based on changes.
    They automatically register event handlers based on method names.

    Handler methods should be named: on_EventTypeName(self, event: EventTypeName)

    Generic type TEventProcessor allows you to specify the type of event processor
    this watchdog works with (e.g., BrowserSession, DatabaseSession, etc.)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # allow non-serializable objects like EventBus/EventProcessor in fields
        extra="forbid",  # dont allow implicit class/instance state, everything must be a properly typed Field or PrivateAttr
        validate_assignment=False,  # avoid re-triggering  __init__ / validators on values on every assignment
        revalidate_instances="never",  # avoid re-triggering __init__ / validators and erasing private attrs
    )

    # Class variables to statically define the list of events relevant to each watchdog
    # (not enforced, just to make it easier to understand the code and debug watchdogs at runtime)
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = []  # Events this watchdog listens to
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []  # Events this watchdog emits

    # Core dependencies
    event_bus: EventBus = Field()
    event_processor: Any = Field()  # Use Any to avoid Pydantic validation issues with generic types

    # Shared state that other watchdogs might need to access should not be defined on EventProcessor, not here!
    # Shared helper methods needed by other watchdogs should be defined on EventProcessor, not here!
    # Alternatively, expose some events on the watchdog to allow access to state/helpers via event_bus system.

    # Private state internal to the watchdog can be defined like this on BaseWatchdog subclasses:
    # _cache: dict[str, bytes] = PrivateAttr(default_factory=dict)
    # _watcher_task: asyncio.Task | None = PrivateAttr(default=None)
    # _download_tasks: WeakSet[asyncio.Task] = PrivateAttr(default_factory=WeakSet)
    # ...

    @property
    def logger(self):
        """Get the logger from the event processor."""
        return self.event_processor.logger

    @staticmethod
    def attach_handler_to_processor(
        event_processor: EventProcessor, event_class: type[BaseEvent[Any]], handler
    ) -> None:
        """Attach a single event handler to an event processor.

        Args:
            event_processor: The event processor to attach to
            event_class: The event class to listen for
            handler: The handler method (must start with 'on_' and end with event type)
        """
        event_bus = event_processor.event_bus

        # Validate handler naming convention
        assert hasattr(handler, "__name__"), "Handler must have a __name__ attribute"
        assert handler.__name__.startswith("on_"), f'Handler {handler.__name__} must start with "on_"'
        assert handler.__name__.endswith(
            event_class.__name__
        ), f"Handler {handler.__name__} must end with event type {event_class.__name__}"

        # Get the watchdog instance if this is a bound method
        watchdog_instance = getattr(handler, "__self__", None)
        watchdog_class_name = watchdog_instance.__class__.__name__ if watchdog_instance else "Unknown"

        # Color codes for logging
        red = "\033[91m"
        green = "\033[92m"
        yellow = "\033[93m"
        magenta = "\033[95m"
        cyan = "\033[96m"
        reset = "\033[0m"

        # Create a wrapper function with unique name to avoid duplicate handler warnings
        # Capture handler by value to avoid closure issues
        def make_unique_handler(actual_handler):
            async def unique_handler(event):
                # just for debug logging, not used for anything else
                parent_event = event_bus.event_history.get(event.event_parent_id) if event.event_parent_id else None
                grandparent_event = (
                    event_bus.event_history.get(parent_event.event_parent_id)
                    if parent_event and parent_event.event_parent_id
                    else None
                )
                parent = (
                    f"{yellow}↲  triggered by {cyan}on_{parent_event.event_type}#{parent_event.event_id[-4:]}{reset}"
                    if parent_event
                    else f"{magenta}👈 by EventProcessor{reset}"
                )
                grandparent = (
                    (
                        f"{yellow}↲  under {cyan}{grandparent_event.event_type}#{grandparent_event.event_id[-4:]}{reset}"
                        if grandparent_event
                        else f"{magenta}👈 by EventProcessor{reset}"
                    )
                    if parent_event
                    else ""
                )
                event_str = f"#{event.event_id[-4:]}"
                time_start = time.time()
                watchdog_and_handler_str = f"[{watchdog_class_name}.{actual_handler.__name__}({event_str})]".ljust(54)
                event_processor.logger.debug(
                    f"{cyan}🚌 {watchdog_and_handler_str} ⏳ Starting...      {reset} {parent} {grandparent}"
                )

                try:
                    # **EXECUTE THE EVENT HANDLER FUNCTION**
                    result = await actual_handler(event)

                    if isinstance(result, Exception):
                        raise result

                    # just for debug logging, not used for anything else
                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    result_summary = "" if result is None else f" ➡️ {magenta}<{type(result).__name__}>{reset}"
                    parents_summary = f" {parent}".replace(
                        "↲  triggered by ", f"⤴  {green}returned to  {cyan}"
                    ).replace("👈 by EventProcessor", f"👉 {green}returned to  {magenta}EventProcessor{reset}")
                    event_processor.logger.debug(
                        f"{green}🚌 {watchdog_and_handler_str} ✅ Succeeded ({time_elapsed:.2f}s){reset}{result_summary}{parents_summary}"
                    )
                    return result
                except Exception as e:
                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    event_processor.logger.error(
                        f"{red}🚌 {watchdog_and_handler_str} ❌ Failed ({time_elapsed:.2f}s): {type(e).__name__}: {e}{reset}"
                    )

                    # Attempt to handle errors - subclasses can override this method
                    try:
                        await watchdog_instance._handle_handler_error(e, event, actual_handler)
                    except Exception as sub_error:
                        event_processor.logger.error(
                            f"{red}🚌 {watchdog_and_handler_str} ❌ Error handling failed: {type(sub_error).__name__}: {sub_error}{reset}"
                        )
                        raise

                    raise

            return unique_handler

        unique_handler = make_unique_handler(handler)
        unique_handler.__name__ = f"{watchdog_class_name}.{handler.__name__}"

        # Check if this handler is already registered - throw error if duplicate
        existing_handlers = event_bus.handlers.get(event_class.__name__, [])
        handler_names = [getattr(h, "__name__", str(h)) for h in existing_handlers]

        if unique_handler.__name__ in handler_names:
            raise RuntimeError(
                f"[{watchdog_class_name}] Duplicate handler registration attempted! "
                f"Handler {unique_handler.__name__} is already registered for {event_class.__name__}. "
                f"This likely means attach_to_processor() was called multiple times."
            )

        event_bus.on(event_class, unique_handler)

    async def _handle_handler_error(self, error: Exception, event: BaseEvent[Any], handler) -> None:
        """Handle errors that occur in event handlers.

        Subclasses can override this method to implement custom error handling logic.
        Default implementation does nothing.

        Args:
            error: The exception that occurred
            event: The event that was being processed
            handler: The handler method that failed
        """

    def attach_to_processor(self) -> None:
        """Attach watchdog to its event processor and start monitoring.

        This method handles event listener registration. The watchdog is already
        bound to an event processor via self.event_processor from initialization.
        """
        # Register event handlers automatically based on method names
        assert self.event_processor is not None, "Event processor not initialized"

        # Find all handler methods (on_EventName)
        registered_events = set()
        for method_name in dir(self):
            if method_name.startswith("on_") and callable(getattr(self, method_name)):
                # Extract event name from method name (on_EventName -> EventName)
                event_name = method_name[3:]  # Remove 'on_' prefix

                # Try to find the event class in the event bus handlers
                event_class = None
                for registered_event_name, handlers in self.event_bus.handlers.items():
                    if registered_event_name == event_name and handlers:
                        # Get the event class from the first handler
                        event_class = type(handlers[0].__annotations__.get("event", BaseEvent))

                if event_class and issubclass(event_class, BaseEvent):
                    # ASSERTION: If LISTENS_TO is defined, enforce it
                    if self.LISTENS_TO:
                        assert event_class in self.LISTENS_TO, (
                            f"[{self.__class__.__name__}] Handler {method_name} listens to {event_name} "
                            f"but {event_name} is not declared in LISTENS_TO: {[e.__name__ for e in self.LISTENS_TO]}"
                        )

                    handler = getattr(self, method_name)

                    # Use the static helper to attach the handler
                    self.attach_handler_to_processor(self.event_processor, event_class, handler)
                    registered_events.add(event_class)

        # ASSERTION: If LISTENS_TO is defined, ensure all declared events have handlers
        if self.LISTENS_TO:
            missing_handlers = set(self.LISTENS_TO) - registered_events
            if missing_handlers:
                missing_names = [e.__name__ for e in missing_handlers]
                self.logger.warning(
                    f"[{self.__class__.__name__}] LISTENS_TO declares {missing_names} "
                    f'but no handlers found (missing on_{"_, on_".join(missing_names)} methods)'
                )

    def emit_event(self, event: BaseEvent[Any]) -> None:
        """Emit an event to the event bus.

        Args:
            event: The event to emit
        """
        if self.EMITS:
            event_type = type(event)
            assert event_type in self.EMITS, (
                f"[{self.__class__.__name__}] Attempting to emit {event_type.__name__} "
                f"but it is not declared in EMITS: {[e.__name__ for e in self.EMITS]}"
            )

        self.event_bus.dispatch(event)

    def __del__(self) -> None:
        """Clean up any running tasks during garbage collection."""

        # A BIT OF MAGIC: Cancel any private attributes that look like asyncio tasks
        try:
            for attr_name in dir(self):
                # e.g. _watcher_task = asyncio.Task
                if attr_name.startswith("_") and attr_name.endswith("_task"):
                    try:
                        task = getattr(self, attr_name)
                        if hasattr(task, "cancel") and callable(task.cancel) and not task.done():
                            task.cancel()
                            # self.logger.debug(f'[{self.__class__.__name__}] Cancelled {attr_name} during cleanup')
                    except Exception:
                        pass  # Ignore errors during cleanup

                # e.g. _download_tasks = WeakSet[asyncio.Task] or list[asyncio.Task]
                if (
                    attr_name.startswith("_")
                    and attr_name.endswith("_tasks")
                    and isinstance(getattr(self, attr_name), Iterable)
                ):
                    for task in getattr(self, attr_name):
                        try:
                            if hasattr(task, "cancel") and callable(task.cancel) and not task.done():
                                task.cancel()
                                # self.logger.debug(f'[{self.__class__.__name__}] Cancelled {attr_name} during cleanup')
                        except Exception:
                            pass  # Ignore errors during cleanup
        except Exception as e:
            # Use a basic logger if available, otherwise ignore
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"⚠️ Error during {self.__class__.__name__} garbage collection __del__(): {type(e)}: {e}"
                    )
            except Exception:
                pass  # Ignore errors during cleanup
