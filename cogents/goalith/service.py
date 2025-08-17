"""
Main service layer for the GoalithService.

Provides a unified facade for all GoalithService functionality.
"""

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Set

from .base.conflict import Conflict, ConflictResolver
from .base.errors import NodeNotFoundError
from .base.goal_node import GoalNode, NodeStatus, NodeType
from .base.graph_store import GraphStore
from .base.memory import MemoryInterface
from .base.update_event import UpdateEvent, UpdateType
from .conflict import AutomaticConflictResolver, ConflictDetector, ConflictOrchestrator
from .decomposer import DecomposerRegistry, default_registry
from .memory import MemoryManager
from .notification import CallableSubscriber, Notifier
from .replan import Replanner, create_default_triggers
from .schedule import Scheduler
from .update import UpdateProcessor, UpdateQueue


class GoalithService:
    """
    Main service class that provides a unified interface to all GoalithService functionality.

    Acts as a facade aggregating all sub-modules and providing high-level methods
    for goal management, decomposition, scheduling, and monitoring.
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        decomposer_registry: Optional[DecomposerRegistry] = None,
        scheduler: Optional[Scheduler] = None,
        memory_backend: Optional[MemoryInterface] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        update_queue_size: int = 1000,
    ):
        """
        Initialize GoalithService.

        Args:
            graph_store: Graph storage backend
            decomposer_registry: Registry for goal decomposers
            scheduler: Task scheduler
            memory_backend: Memory storage backend
            conflict_resolver: Conflict resolution strategy
            update_queue_size: Maximum size of update queue
        """
        # Core components
        self._graph_store = graph_store or GraphStore()
        self._decomposer_registry = decomposer_registry or default_registry
        self._scheduler = scheduler or Scheduler()
        self._memory_manager = MemoryManager(memory_backend)

        # Update system
        self._update_queue = UpdateQueue(maxsize=update_queue_size)
        self._notifier = Notifier()

        # Conflict management
        self._conflict_detector = ConflictDetector(self._graph_store)
        self._conflict_orchestrator = ConflictOrchestrator(
            self._conflict_detector, conflict_resolver or AutomaticConflictResolver()
        )

        # Update processor with conflict integration
        self._update_processor = UpdateProcessor(self._graph_store, self._conflict_orchestrator)

        # Replanning system
        self._replanner = Replanner(self._graph_store, self._decomposer_registry, self._scheduler)

        # Add default triggers
        for trigger in create_default_triggers():
            self._replanner.add_trigger(trigger)

        # Processing control
        self._processing = False
        self._process_thread: Optional[threading.Thread] = None
        self._process_queue = Queue()

        # Statistics
        self._stats = {
            "goals_created": 0,
            "decompositions": 0,
            "tasks_scheduled": 0,
            "updates_processed": 0,
            "conflicts_resolved": 0,
            "replans_executed": 0,
        }

    # Core goal management methods

    def create_goal(
        self,
        description: str,
        goal_type: NodeType = NodeType.GOAL,
        priority: float = 1.0,
        deadline: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enrich_context: bool = True,
    ) -> str:
        """
        Create a new goal in the system.

        Args:
            description: Description of the goal
            goal_type: Type of goal (GOAL, SUBGOAL, TASK)
            priority: Priority level
            deadline: Optional deadline
            context: Additional context data
            tags: Tags for categorization
            enrich_context: Whether to enrich context with memory data

        Returns:
            ID of the created goal
        """
        goal = GoalNode(
            description=description,
            type=goal_type,
            priority=priority,
            deadline=deadline,
            context=context or {},
            tags=tags or [],
        )

        # Enrich with memory context if requested
        if enrich_context:
            enriched_context = self._memory_manager.enrich_node_context(goal)
            goal.context.update(enriched_context)

        # Add to graph
        self._graph_store.add_node(goal)

        # Notify subscribers
        self._notify_update(
            UpdateEvent(
                update_type=UpdateType.NODE_ADD,
                node_id=goal.id,
                data={"node": goal.model_dump(mode="json")},
                source="goalith_service",
            )
        )

        self._stats["goals_created"] += 1
        return goal.id

    # Goal management aliases and methods expected by tests
    def get_goal(self, goal_id: str) -> Optional[GoalNode]:
        """Alias for get_node that returns None if node doesn't exist."""
        try:
            return self.get_node(goal_id)
        except NodeNotFoundError:
            return None

    def update_goal(self, goal_id: str, **updates) -> bool:
        """Update a goal with new values."""
        try:
            goal = self.get_node(goal_id)
            for key, value in updates.items():
                if hasattr(goal, key):
                    setattr(goal, key, value)
            goal.update_context("last_updated", datetime.now(timezone.utc).isoformat())
            self._graph_store.update_node(goal)
            self._notify_update(
                UpdateEvent(
                    update_type=UpdateType.NODE_EDIT,
                    node_id=goal_id,
                    data={"updates": updates},
                    source="goalith_service",
                )
            )
            return True
        except Exception:
            return False

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal from the system."""
        try:
            self._graph_store.remove_node(goal_id)
            self._notify_update(
                UpdateEvent(
                    update_type=UpdateType.NODE_REMOVE,
                    node_id=goal_id,
                    data={},
                    source="goalith_service",
                )
            )
            return True
        except Exception:
            return False

    def list_goals(self, status: Optional[NodeStatus] = None) -> List[GoalNode]:
        """List all goals, optionally filtered by status."""
        if status:
            return self.get_nodes_by_status(status)
        return self._graph_store.get_all_nodes()

    def add_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """Add a dependency relationship between goals."""
        try:
            self._graph_store.add_dependency(dependent_id, dependency_id)
            self._notify_update(
                UpdateEvent(
                    update_type=UpdateType.DEPENDENCY_ADD,
                    node_id=dependent_id,
                    data={"dependency_id": dependency_id},
                    source="goalith_service",
                )
            )
            return True
        except Exception:
            return False

    def remove_dependency(self, dependency_id: str, dependent_id: str) -> bool:
        """Remove a dependency relationship between goals."""
        try:
            self._graph_store.remove_dependency(dependent_id, dependency_id)
            self._notify_update(
                UpdateEvent(
                    update_type=UpdateType.DEPENDENCY_REMOVE,
                    node_id=dependent_id,
                    data={"dependency_id": dependency_id},
                    source="goalith_service",
                )
            )
            return True
        except Exception:
            return False

    def decompose_goal(
        self,
        goal_id: str,
        decomposer_name: str = "llm_decomposer",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Decompose a goal into subgoals or tasks.

        Args:
            goal_id: ID of the goal to decompose
            decomposer_name: Name of the decomposer to use
            context: Additional context for decomposition

        Returns:
            Dictionary containing list of subgoal IDs or error information
        """
        # Check if goal exists
        if not self._graph_store.has_node(goal_id):
            return {"error": f"Goal {goal_id} not found"}

        try:
            goal = self._graph_store.get_node(goal_id)

            # Perform decomposition
            subgoals = self._decomposer_registry.decompose(decomposer_name, goal, context)
        except (KeyError, Exception) as e:
            return {"error": str(e)}

        # Add subgoals to graph
        subgoal_ids = []
        for subgoal in subgoals:
            # Enrich with memory context
            enriched_context = self._memory_manager.enrich_node_context(subgoal)
            subgoal.context.update(enriched_context)

            self._graph_store.add_node(subgoal)
            self._graph_store.add_parent_child(goal_id, subgoal.id)
            # Subgoal depends on parent goal
            self._graph_store.add_dependency(goal_id, subgoal.id)
            subgoal_ids.append(subgoal.id)

            # Notify about new subgoal
            self._notify_update(
                UpdateEvent(
                    update_type=UpdateType.NODE_ADD,
                    node_id=subgoal.id,
                    data={"node": subgoal.model_dump(mode="json")},
                    source="goalith_service",
                )
            )

        # Update the parent goal
        self._graph_store.update_node(goal)

        self._stats["decompositions"] += 1
        return {"subgoal_ids": subgoal_ids}

    def get_next_task(self, criteria: Optional[Dict[str, Any]] = None) -> Optional[GoalNode]:
        """
        Get the next highest-priority ready task.

        Args:
            criteria: Optional filtering criteria

        Returns:
            Next task to execute, or None if none available
        """
        ready_nodes = self._graph_store.get_ready_nodes()

        if criteria:
            next_task = self._scheduler.get_next_with_criteria(ready_nodes, criteria)
        else:
            next_task = self._scheduler.get_next(ready_nodes)

        if next_task:
            self._stats["tasks_scheduled"] += 1

        return next_task

    def get_ready_tasks(self, limit: Optional[int] = None) -> List[GoalNode]:
        """
        Get all ready tasks, optionally limited and sorted by priority.

        Args:
            limit: Optional limit on number of tasks

        Returns:
            List of ready tasks
        """
        ready_nodes = self._graph_store.get_ready_nodes()
        sorted_nodes = self._scheduler.peek_all(ready_nodes)

        if limit:
            return sorted_nodes[:limit]
        return sorted_nodes

    # Update and notification methods

    def post_update(
        self,
        update_type: UpdateType,
        node_id: str,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> bool:
        """
        Post an update event for processing.

        Args:
            update_type: Type of update
            node_id: ID of affected node
            data: Update data
            source: Source of the update

        Returns:
            True if update was queued successfully
        """
        event = UpdateEvent(update_type=update_type, node_id=node_id, data=data or {}, source=source)

        try:
            self._update_queue.put(event, block=False)
            return True
        except:
            return False

    def subscribe(
        self,
        callback: callable,
        subscriber_id: str,
        event_types: Optional[Set[UpdateType]] = None,
        node_ids: Optional[Set[str]] = None,
    ) -> None:
        """
        Subscribe to notifications.

        Args:
            callback: Function to call on notifications
            subscriber_id: Unique subscriber ID
            event_types: Optional filter for event types
            node_ids: Optional filter for specific nodes
        """
        subscriber = CallableSubscriber(callback, subscriber_id)
        self._notifier.subscribe(subscriber, event_types, node_ids)

    def unsubscribe(self, subscriber_id: str) -> None:
        """
        Unsubscribe from notifications.

        Args:
            subscriber_id: ID of subscriber to remove
        """
        self._notifier.unsubscribe(subscriber_id)

    # Conflict and replanning methods

    def trigger_replan(
        self,
        trigger_type: str = "manual_trigger",
        affected_nodes: Optional[List[str]] = None,
        scope: str = "local",
        description: str = "Manual replan trigger",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Manually trigger replanning.

        Args:
            trigger_type: Type of trigger
            affected_nodes: Nodes to replan
            scope: Scope of replanning
            description: Description of the replan
            metadata: Additional metadata

        Returns:
            True if replanning was successful
        """
        from .base.replanning import ReplanEvent, ReplanScope, TriggerType

        replan_event = ReplanEvent(
            trigger_type=TriggerType(trigger_type),
            affected_nodes=affected_nodes or [],
            scope=ReplanScope(scope),
            description=description,
            metadata=metadata or {},
        )

        success = self._replanner.execute_replan(replan_event)
        if success:
            self._stats["replans_executed"] += 1

        return success

    def get_conflicts(self) -> List[Conflict]:
        """
        Get current conflicts in the system.

        Returns:
            List of current conflicts
        """
        # Check for conflicts in recent updates
        recent_events = self._update_queue.get_all()
        return self._conflict_orchestrator.detect_conflicts(recent_events)

    # Query and monitoring methods

    def get_node(self, node_id: str) -> GoalNode:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            The node

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        return self._graph_store.get_node(node_id)

    def get_children(self, node_id: str) -> List[GoalNode]:
        """
        Get child nodes of a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child nodes
        """
        return self._graph_store.get_children(node_id)

    def get_dependencies(self, node_id: str) -> List[GoalNode]:
        """
        Get dependency nodes of a given node.

        Args:
            node_id: Dependent node ID

        Returns:
            List of dependency nodes
        """
        return self._graph_store.get_dependencies(node_id)

    def get_nodes_by_status(self, status: NodeStatus) -> List[GoalNode]:
        """
        Get all nodes with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of nodes with the given status
        """
        return self._graph_store.get_nodes_by_status(status)

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            System statistics
        """
        graph_stats = self._graph_store.get_graph_stats()
        scheduling_stats = self._scheduler.get_scheduling_stats(self.get_ready_tasks())
        memory_stats = self._memory_manager.get_memory_stats()
        conflict_stats = self._conflict_orchestrator.get_system_stats()
        replanning_stats = self._replanner.get_replanning_stats()
        update_stats = self._update_processor.get_processing_stats()

        return {
            "service": self._stats.copy(),
            "graph": graph_stats,
            "scheduling": scheduling_stats,
            "memory": memory_stats,
            "conflicts": conflict_stats,
            "replanning": replanning_stats,
            "updates": update_stats,
            "queue_size": self._update_queue.size(),
            "subscribers": self._notifier.get_subscriber_count(),
        }

    # Processing control methods

    def start_processing(self) -> None:
        """Start background processing of updates."""
        if self._processing:
            return

        self._processing = True
        self._process_thread = threading.Thread(target=self._process_updates, daemon=True)
        self._process_thread.start()

    def stop_processing(self) -> None:
        """Stop background processing of updates."""
        self._processing = False
        if self._process_thread:
            self._process_thread.join(timeout=5.0)
            self._process_thread = None

    def process_pending_updates(self) -> int:
        """
        Process all pending updates synchronously.

        Returns:
            Number of updates processed
        """
        processed = 0
        events = self._update_queue.get_all()

        for event in events:
            if self._process_single_update(event):
                processed += 1

        return processed

    def _process_updates(self) -> None:
        """Background update processing loop."""
        while self._processing:
            try:
                event = self._update_queue.get(timeout=1.0)
                self._process_single_update(event)
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing update: {e}")

    def _process_single_update(self, event: UpdateEvent) -> bool:
        """
        Process a single update event.

        Args:
            event: The update event

        Returns:
            True if processed successfully
        """
        try:
            # Check for conflicts
            (
                conflicts,
                all_resolved,
            ) = self._conflict_orchestrator.process_with_conflict_check([event])

            if conflicts and not all_resolved:
                print(f"Unresolved conflicts for event {event.id}: {len(conflicts)}")
                return False

            # Process the update
            success = self._update_processor.process_event(event)

            if success:
                # Check for replanning triggers
                replan_events = self._replanner.check_triggers(event)

                # Execute replanning if triggered
                for replan_event in replan_events:
                    self._replanner.execute_replan(replan_event)
                    self._stats["replans_executed"] += 1

                # Notify subscribers
                self._notifier.notify(event)
                self._stats["updates_processed"] += 1

            return success

        except Exception as e:
            print(f"Error processing update event {event.id}: {e}")
            return False

    def _notify_update(self, event: UpdateEvent) -> None:
        """Notify about an update event."""
        self._notifier.notify(event)

    # Persistence methods

    def save_to_file(self, filepath: Path) -> bool:
        """
        Save the current state to a file.

        Args:
            filepath: Path to save to

        Returns:
            True if successful
        """
        try:
            self._graph_store.save_to_file(filepath)
            return True
        except Exception as e:
            print(f"Error saving to file: {e}")
            return False

    def load_from_file(self, filepath: Path) -> bool:
        """
        Load state from a file.

        Args:
            filepath: Path to load from

        Returns:
            True if successful
        """
        try:
            self._graph_store.load_from_file(filepath)
            return True
        except Exception as e:
            print(f"Error loading from file: {e}")
            return False

    # Context and memory methods

    def get_enriched_goal(self, goal_id: str) -> Optional[GoalNode]:
        """Get an enriched goal with memory context."""
        try:
            goal = self._graph_store.get_node(goal_id)
            if goal:
                # Enrich the goal with memory context using the default context key
                enriched_goal = self._memory_manager.enrich_goal(goal, "context")
                return enriched_goal
            return None
        except Exception:
            return None

    def search_related_goals(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for goals related to a query."""
        return self._memory_manager.search_related_goals(query, limit)

    def store_goal_context(self, goal_id: str, context: Dict[str, Any]) -> bool:
        """Store context for a goal by ID."""
        return self._memory_manager.store_goal_context(goal_id, context)

    def store_node_context(self, node: GoalNode, context_key: str, context_value: Any) -> bool:
        """Store context for a goal node (original method)."""
        return self._memory_manager.store_node_context(node, context_key, context_value)

    def get_node_context(self, node_id: str, context_key: str) -> Optional[Any]:
        """
        Get stored context for a node.

        Args:
            node_id: Node ID
            context_key: Context key

        Returns:
            Context value or None
        """
        if not self._graph_store.has_node(node_id):
            return None

        node = self._graph_store.get_node(node_id)
        return self._memory_manager.get_node_context(node, context_key)

    def get_goal_context(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a goal by ID."""
        return self._memory_manager.get_goal_context(goal_id)

    def get_similar_nodes(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find nodes similar to the given node.

        Args:
            node_id: Reference node ID
            limit: Maximum results

        Returns:
            List of similar nodes
        """
        if not self._graph_store.has_node(node_id):
            return []

        node = self._graph_store.get_node(node_id)
        return self._memory_manager.get_similar_nodes(node, limit)

    # Cleanup and maintenance

    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up old data from the system.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Cleanup statistics
        """

        max_age = timedelta(days=max_age_days)

        # Clean memory data
        memory_cleaned = self._memory_manager.cleanup_old_data(max_age)

        # Could add graph cleanup here (remove completed nodes older than threshold)
        graph_cleaned = 0

        return {
            "memory_items_cleaned": memory_cleaned,
            "graph_items_cleaned": graph_cleaned,
            "total_cleaned": memory_cleaned + graph_cleaned,
        }

    # Utility methods for external integrations

    def get_decomposer_registry(self) -> DecomposerRegistry:
        """Get the decomposer registry for external registration."""
        return self._decomposer_registry

    def get_scheduler(self) -> Scheduler:
        """Get the scheduler for external policy configuration."""
        return self._scheduler

    def get_memory_manager(self) -> MemoryManager:
        """Get the memory manager for external operations."""
        return self._memory_manager

    def set_conflict_resolver(self, resolver: ConflictResolver) -> None:
        """Set a custom conflict resolver."""
        self._conflict_orchestrator.set_resolver(resolver)

    def add_replan_trigger(self, trigger) -> None:
        """Add a custom replan trigger."""
        self._replanner.add_trigger(trigger)

    # Additional methods expected by tests
    def schedule_tasks(self, limit: Optional[int] = None) -> List[GoalNode]:
        """Schedule tasks using the scheduler."""
        ready_tasks = self.get_ready_tasks(limit)
        # The scheduler would typically arrange these tasks
        return ready_tasks

    def process_updates(self) -> int:
        """Process pending updates in the queue."""
        return self.process_pending_updates()

    def subscribe_to_updates(self, callback) -> str:
        """Subscribe to update notifications."""
        subscriber_id = f"auto_{datetime.now().timestamp()}"
        self.subscribe(callback, subscriber_id)
        return subscriber_id

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.get_system_stats()
        # Flatten service stats to top level for backward compatibility
        service_stats = stats.pop("service", {})
        stats.update(service_stats)
        return stats

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        system_stats = self.get_system_stats()
        memory_stats = self._memory_manager.get_memory_stats()

        # Get total goals from graph store
        total_goals = 0
        try:
            total_goals = len(self._graph_store.get_all_nodes())
        except:
            total_goals = 0

        return {
            "system_stats": system_stats,
            "memory_stats": memory_stats,
            "queue_size": self._update_queue.size(),
            "total_goals": total_goals,
            "processing_time": system_stats.get("uptime", 0),  # Use uptime as processing time
        }
