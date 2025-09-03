"""Example demonstrating the generic watchdog framework."""

import asyncio
import logging

from bubus import BaseEvent, EventBus
from bubus.models import T_EventResultType
from pydantic import Field

from cogents.base.buspark import BaseWatchdog


# Example event classes
class UserLoginEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a user logs in."""

    user_id: str = Field()
    timestamp: float = Field()


class UserLogoutEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a user logs out."""

    user_id: str = Field()
    timestamp: float = Field()


class SecurityAlertEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a security alert is triggered."""

    alert_type: str = Field()
    user_id: str = Field()
    severity: str = Field()


# Example event processor (could be a database session, API client, etc.)
class DatabaseSession:
    """Example event processor that manages database operations."""

    def __init__(self, name: str):
        self.name = name
        self.event_bus = EventBus()
        self.logger = logging.getLogger(f"DatabaseSession.{name}")
        self.active_users = set()

    async def login_user(self, user_id: str):
        """Simulate user login."""
        self.active_users.add(user_id)
        self.logger.info(f"User {user_id} logged in")

        # Emit login event
        event = UserLoginEvent(user_id=user_id, timestamp=asyncio.get_event_loop().time())
        self.event_bus.dispatch(event)

    async def logout_user(self, user_id: str):
        """Simulate user logout."""
        if user_id in self.active_users:
            self.active_users.remove(user_id)
            self.logger.info(f"User {user_id} logged out")

            # Emit logout event
            event = UserLogoutEvent(user_id=user_id, timestamp=asyncio.get_event_loop().time())
            self.event_bus.dispatch(event)


# Example watchdog that monitors user activity
class UserActivityWatchdog(BaseWatchdog[DatabaseSession]):
    """Watchdog that monitors user activity and emits security alerts."""

    # Declare which events this watchdog listens to
    LISTENS_TO = [UserLoginEvent, UserLogoutEvent]

    # Declare which events this watchdog emits
    EMITS = [SecurityAlertEvent]

    def __init__(self, event_bus: EventBus, event_processor: DatabaseSession):
        super().__init__(event_bus=event_bus, event_processor=event_processor)
        self._login_count = 0
        self._logout_count = 0

    async def on_UserLoginEvent(self, event: UserLoginEvent):
        """Handle user login events."""
        self._login_count += 1
        self.logger.info(f"User {event.user_id} logged in (total logins: {self._login_count})")

        # Check for suspicious activity (multiple logins in short time)
        if self._login_count > 5:
            alert = SecurityAlertEvent(alert_type="multiple_logins", user_id=event.user_id, severity="medium")
            self.emit_event(alert)
            self.logger.warning(f"Security alert: Multiple logins detected for user {event.user_id}")

    async def on_UserLogoutEvent(self, event: UserLogoutEvent):
        """Handle user logout events."""
        self._logout_count += 1
        self.logger.info(f"User {event.user_id} logged out (total logins: {self._logout_count})")

        # Check for unusual logout patterns
        if self._logout_count > 10:
            alert = SecurityAlertEvent(alert_type="frequent_logouts", user_id=event.user_id, severity="low")
            self.emit_event(alert)
            self.logger.warning(f"Security alert: Frequent logouts detected for user {event.user_id}")


# Example watchdog that logs all events
class EventLoggerWatchdog(BaseWatchdog[DatabaseSession]):
    """Watchdog that logs all events for debugging purposes."""

    # Listen to all events (empty list means listen to everything)
    LISTENS_TO = []

    # Don't emit any events
    EMITS = []

    def __init__(self, event_bus: EventBus, event_processor: DatabaseSession):
        super().__init__(event_bus=event_bus, event_processor=event_processor)

    async def on_UserLoginEvent(self, event: UserLoginEvent):
        """Log user login events."""
        self.logger.info(f"🔐 LOGIN: User {event.user_id} at {event.timestamp}")

    async def on_UserLogoutEvent(self, event: UserLogoutEvent):
        """Log user logout events."""
        self.logger.info(f"🚪 LOGOUT: User {event.user_id} at {event.timestamp}")

    async def on_SecurityAlertEvent(self, event: SecurityAlertEvent):
        """Log security alert events."""
        self.logger.warning(f"🚨 ALERT: {event.alert_type} for user {event.user_id} - {event.severity}")


async def main():
    """Main example function."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create event bus
        event_bus = EventBus()

        # Create database session
        db_session = DatabaseSession("main_db")

        # Create watchdogs
        user_watchdog = UserActivityWatchdog(event_bus=event_bus, event_processor=db_session)
        logger_watchdog = EventLoggerWatchdog(event_bus=event_bus, event_processor=db_session)

        # Attach watchdogs to the processor
        user_watchdog.attach_to_processor()
        logger_watchdog.attach_to_processor()

        # Simulate user activity
        print("🚀 Starting user activity simulation...")

        # Simulate multiple logins
        for i in range(7):
            await db_session.login_user(f"user_{i}")
            await asyncio.sleep(0.1)  # Small delay

        # Simulate some logouts
        for i in range(3):
            await db_session.logout_user(f"user_{i}")
            await asyncio.sleep(0.1)

        # Simulate more logins to trigger security alerts
        for i in range(5):
            await db_session.login_user(f"user_{i+10}")
            await asyncio.sleep(0.1)

        print("✅ User activity simulation completed!")

        # Wait for all events to be processed
        print("⏳ Waiting for all events to complete...")
        await event_bus.wait_until_idle(timeout=10.0)
        print("✅ All events completed!")

        # Try to stop the event bus gracefully
        print("🛑 Stopping event bus...")
        try:
            # Use a reasonable timeout for graceful shutdown
            await asyncio.wait_for(event_bus.stop(timeout=5.0, clear=True), timeout=10.0)
            print("✅ Event bus stopped gracefully!")
        except asyncio.TimeoutError:
            print("⚠️  Event bus stop timed out, forcing shutdown...")
            # Force shutdown by setting internal flags
            if hasattr(event_bus, "_is_running"):
                event_bus._is_running = False
            if hasattr(event_bus, "event_queue") and event_bus.event_queue:
                event_bus.event_queue.shutdown(immediate=True)

        # Wait a bit for any remaining tasks to finish
        print("🧹 Waiting for cleanup...")
        await asyncio.sleep(1.0)

        # Check remaining tasks and force cleanup if needed
        remaining_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if remaining_tasks:
            print(f"⚠️  {len(remaining_tasks)} tasks still running, forcing cleanup...")

            # First, try to cancel all tasks
            for task in remaining_tasks:
                if not task.done():
                    task.cancel()

            # Wait a bit for cancellation to take effect
            await asyncio.sleep(0.5)

            # Check which tasks are still running
            still_running = [task for task in remaining_tasks if not task.done()]
            if still_running:
                print(f"⚠️  {len(still_running)} tasks still running after cancellation:")
                for task in still_running:
                    task_name = task.get_name() if hasattr(task, "get_name") else "unnamed"
                    print(f"   - {task_name}")

                # Try one more time with a shorter timeout
                try:
                    await asyncio.wait_for(asyncio.gather(*still_running, return_exceptions=True), timeout=2.0)
                    print("✅ Remaining tasks completed")
                except asyncio.TimeoutError:
                    print("⚠️  Some tasks still hanging, but main function will exit")
            else:
                print("✅ All tasks cleaned up")
        else:
            print("✅ No remaining tasks")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("🏁 Main function completed")


if __name__ == "__main__":
    asyncio.run(main())
