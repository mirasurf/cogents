"""
Unit tests for goalith.service module.
"""
import threading
import time
from unittest.mock import Mock, patch

import pytest

from cogents.goalith.base.errors import NodeNotFoundError
from cogents.goalith.base.goal_node import GoalNode, NodeStatus, NodeType
from cogents.goalith.base.graph_store import GraphStore
from cogents.goalith.base.update_event import UpdateEvent, UpdateType
from cogents.goalith.decomposer.registry import DecomposerRegistry
from cogents.goalith.decomposer.simple_decomposer import SimpleListDecomposer
from cogents.goalith.memory.inmemstore import InMemoryStore
from cogents.goalith.service import GoalithService


class TestGoalithService:
    """Test GoalithService functionality."""

    def test_default_initialization(self):
        """Test creating GoalithService with default components."""
        service = GoalithService()

        assert service._graph_store is not None
        assert service._decomposer_registry is not None
        assert service._scheduler is not None
        assert service._memory_manager is not None
        assert service._update_queue is not None
        assert service._update_processor is not None
        assert service._notifier is not None
        assert service._conflict_orchestrator is not None
        assert service._replanner is not None
        assert service._background_thread is None
        assert service._shutdown_event is not None

    def test_custom_initialization(self):
        """Test creating GoalithService with custom components."""
        custom_graph = GraphStore()
        custom_registry = DecomposerRegistry()
        custom_memory = InMemoryStore()

        service = GoalithService(
            graph_store=custom_graph,
            decomposer_registry=custom_registry,
            memory_backend=custom_memory,
            update_queue_size=500,
        )

        assert service._graph_store is custom_graph
        assert service._decomposer_registry is custom_registry
        assert service._memory_manager._backend is custom_memory

    def test_add_goal(self):
        """Test adding a goal to the service."""
        service = GoalithService()

        goal = service.add_goal(
            description="Test goal", goal_type=NodeType.GOAL, priority=7.0, tags=["test"], metadata={"project": "test"}
        )

        assert isinstance(goal, GoalNode)
        assert goal.description == "Test goal"
        assert goal.type == NodeType.GOAL
        assert goal.priority == 7.0
        assert "test" in goal.tags
        assert goal.metadata["project"] == "test"

        # Verify it's in the graph
        assert service._graph_store.has_node(goal.id)

    def test_add_goal_with_dependencies(self):
        """Test adding a goal with dependencies."""
        service = GoalithService()

        # Add dependency first
        dep_goal = service.add_goal("Dependency goal")

        # Add goal with dependency
        goal = service.add_goal(description="Main goal", dependencies=[dep_goal.id])

        assert dep_goal.id in goal.dependencies

        # Verify dependency in graph
        parents = service._graph_store.get_parents(goal.id)
        parent_ids = {p.id for p in parents}
        assert dep_goal.id in parent_ids

    def test_add_goal_with_invalid_dependency_raises_error(self):
        """Test that adding a goal with invalid dependency raises error."""
        service = GoalithService()

        with pytest.raises(NodeNotFoundError):
            service.add_goal(description="Goal with bad dependency", dependencies=["nonexistent-id"])

    def test_get_goal(self):
        """Test retrieving a goal by ID."""
        service = GoalithService()

        original = service.add_goal("Test goal")
        retrieved = service.get_goal(original.id)

        assert retrieved == original

    def test_get_nonexistent_goal_raises_error(self):
        """Test that getting nonexistent goal raises error."""
        service = GoalithService()

        with pytest.raises(NodeNotFoundError):
            service.get_goal("nonexistent")

    def test_update_goal(self):
        """Test updating a goal."""
        service = GoalithService()

        goal = service.add_goal("Original description")

        updated = service.update_goal(
            goal.id, description="Updated description", priority=8.0, status=NodeStatus.IN_PROGRESS
        )

        assert updated.description == "Updated description"
        assert updated.priority == 8.0
        assert updated.status == NodeStatus.IN_PROGRESS

    def test_update_nonexistent_goal_raises_error(self):
        """Test that updating nonexistent goal raises error."""
        service = GoalithService()

        with pytest.raises(NodeNotFoundError):
            service.update_goal("nonexistent", description="New desc")

    def test_remove_goal(self):
        """Test removing a goal."""
        service = GoalithService()

        goal = service.add_goal("Goal to remove")
        goal_id = goal.id

        # Verify it exists
        assert service._graph_store.has_node(goal_id)

        # Remove it
        service.remove_goal(goal_id)

        # Verify it's gone
        assert not service._graph_store.has_node(goal_id)

    def test_remove_nonexistent_goal_raises_error(self):
        """Test that removing nonexistent goal raises error."""
        service = GoalithService()

        with pytest.raises(NodeNotFoundError):
            service.remove_goal("nonexistent")

    def test_add_dependency(self):
        """Test adding a dependency between goals."""
        service = GoalithService()

        goal1 = service.add_goal("Goal 1")
        goal2 = service.add_goal("Goal 2")

        service.add_dependency(goal1.id, goal2.id)

        # Check dependency was added
        updated_goal1 = service.get_goal(goal1.id)
        assert goal2.id in updated_goal1.dependencies

    def test_remove_dependency(self):
        """Test removing a dependency between goals."""
        service = GoalithService()

        goal1 = service.add_goal("Goal 1")
        goal2 = service.add_goal("Goal 2")

        service.add_dependency(goal1.id, goal2.id)

        # Verify dependency exists
        updated_goal1 = service.get_goal(goal1.id)
        assert goal2.id in updated_goal1.dependencies

        # Remove dependency
        service.remove_dependency(goal1.id, goal2.id)

        # Verify dependency is gone
        updated_goal1 = service.get_goal(goal1.id)
        assert goal2.id not in updated_goal1.dependencies

    def test_get_ready_goals(self):
        """Test getting ready goals."""
        service = GoalithService()

        # Add some goals with dependencies
        goal1 = service.add_goal("Independent goal", status=NodeStatus.PENDING)
        goal2 = service.add_goal("Dependent goal", status=NodeStatus.PENDING)
        goal3 = service.add_goal("Completed dependency", status=NodeStatus.COMPLETED)

        service.add_dependency(goal2.id, goal3.id)  # goal2 depends on completed goal3

        ready = service.get_ready_goals()
        ready_ids = {g.id for g in ready}

        # Both goal1 (independent) and goal2 (depends on completed goal3) should be ready
        assert goal1.id in ready_ids
        assert goal2.id in ready_ids

    def test_get_next_goal(self):
        """Test getting the next highest priority goal."""
        service = GoalithService()

        # Add goals with different priorities
        low_priority = service.add_goal("Low priority", priority=1.0)
        high_priority = service.add_goal("High priority", priority=9.0)
        medium_priority = service.add_goal("Medium priority", priority=5.0)

        next_goal = service.get_next_goal()

        # Should return highest priority goal
        assert next_goal.id == high_priority.id

    def test_get_next_goal_empty_returns_none(self):
        """Test that getting next goal from empty service returns None."""
        service = GoalithService()

        next_goal = service.get_next_goal()
        assert next_goal is None

    def test_list_goals(self):
        """Test listing all goals."""
        service = GoalithService()

        goal1 = service.add_goal("Goal 1")
        goal2 = service.add_goal("Goal 2")
        goal3 = service.add_goal("Goal 3")

        all_goals = service.list_goals()
        goal_ids = {g.id for g in all_goals}

        expected_ids = {goal1.id, goal2.id, goal3.id}
        assert goal_ids == expected_ids

    def test_decompose_goal(self):
        """Test goal decomposition."""
        service = GoalithService()

        # Register a simple decomposer
        decomposer = SimpleListDecomposer()
        service._decomposer_registry.register("simple", decomposer)

        goal = service.add_goal("Main goal")

        # Mock the decomposer to return specific subgoals
        with patch.object(decomposer, "decompose") as mock_decompose:
            mock_decompose.return_value = [
                GoalNode(description="Subgoal 1", type=NodeType.SUBGOAL),
                GoalNode(description="Subgoal 2", type=NodeType.SUBGOAL),
            ]

            subgoals = service.decompose_goal(goal.id, "simple")

            assert len(subgoals) == 2
            assert all(sg.type == NodeType.SUBGOAL for sg in subgoals)

            # Verify subgoals are in graph and depend on main goal
            for subgoal in subgoals:
                assert service._graph_store.has_node(subgoal.id)
                assert goal.id in subgoal.dependencies

    def test_decompose_nonexistent_goal_raises_error(self):
        """Test that decomposing nonexistent goal raises error."""
        service = GoalithService()

        with pytest.raises(NodeNotFoundError):
            service.decompose_goal("nonexistent", "simple")

    def test_start_background_processing(self):
        """Test starting background processing."""
        service = GoalithService()

        assert service._background_thread is None

        service.start_background_processing()

        assert service._background_thread is not None
        assert service._background_thread.is_alive()

        # Clean up
        service.stop_background_processing()

    def test_stop_background_processing(self):
        """Test stopping background processing."""
        service = GoalithService()

        service.start_background_processing()
        assert service._background_thread.is_alive()

        service.stop_background_processing()

        # Give it a moment to shut down
        time.sleep(0.1)
        assert not service._background_thread.is_alive()

    def test_double_start_background_processing(self):
        """Test that starting background processing twice doesn't create multiple threads."""
        service = GoalithService()

        service.start_background_processing()
        first_thread = service._background_thread

        # Try to start again
        service.start_background_processing()
        second_thread = service._background_thread

        # Should be the same thread
        assert first_thread is second_thread

        # Clean up
        service.stop_background_processing()

    def test_context_manager(self):
        """Test using GoalithService as a context manager."""
        with GoalithService() as service:
            assert service._background_thread is not None
            assert service._background_thread.is_alive()

            # Add a goal to verify service is working
            goal = service.add_goal("Test goal")
            assert service._graph_store.has_node(goal.id)

        # Background processing should be stopped after exiting context
        time.sleep(0.1)
        assert not service._background_thread.is_alive()

    def test_get_statistics(self):
        """Test getting service statistics."""
        service = GoalithService()

        # Add some goals
        service.add_goal("Goal 1", status=NodeStatus.PENDING)
        service.add_goal("Goal 2", status=NodeStatus.COMPLETED)
        service.add_goal("Goal 3", status=NodeStatus.IN_PROGRESS)

        stats = service.get_statistics()

        assert isinstance(stats, dict)
        assert "total_nodes" in stats
        assert stats["total_nodes"] == 3
        assert "status_counts" in stats
        assert stats["status_counts"]["pending"] == 1
        assert stats["status_counts"]["completed"] == 1
        assert stats["status_counts"]["in_progress"] == 1

    def test_subscribe_to_notifications(self):
        """Test subscribing to notifications."""
        service = GoalithService()

        # Create a mock subscriber
        subscriber = Mock()

        service.subscribe_to_notifications(subscriber)

        # Verify subscriber was added to notifier
        assert subscriber in service._notifier._subscribers

    def test_queue_update(self):
        """Test queuing an update event."""
        service = GoalithService()

        goal = service.add_goal("Test goal")

        update = UpdateEvent(
            update_type=UpdateType.STATUS_CHANGE,
            node_id=goal.id,
            data={"old_status": "pending", "new_status": "in_progress"},
        )

        # Queue should be empty initially
        assert service._update_queue.size() == 0

        service.queue_update(update)

        # Update should be in queue
        assert service._update_queue.size() == 1

    @patch("cogents.goalith.service.Queue")
    def test_queue_update_when_full(self, mock_queue_class):
        """Test queuing update when queue is full."""
        # Mock a full queue
        mock_queue = Mock()
        mock_queue.put.side_effect = Exception("Queue full")
        mock_queue_class.return_value = mock_queue

        service = GoalithService()

        update = UpdateEvent(update_type=UpdateType.STATUS_CHANGE, node_id="test", data={})

        # Should handle the exception gracefully
        service.queue_update(update)

    def test_process_updates_manually(self):
        """Test manually processing updates."""
        service = GoalithService()

        goal = service.add_goal("Test goal", status=NodeStatus.PENDING)

        # Queue an update
        update = UpdateEvent(
            update_type=UpdateType.STATUS_CHANGE,
            node_id=goal.id,
            data={"old_status": "pending", "new_status": "completed"},
        )
        service.queue_update(update)

        # Process updates manually
        processed_count = service.process_updates()

        assert processed_count == 1
        assert service._update_queue.size() == 0

    def test_memory_integration(self):
        """Test integration with memory system."""
        service = GoalithService()

        goal = service.add_goal("Test goal with context")

        # Store some context
        context = {"domain": "test", "importance": "high"}
        service._memory_manager.store_context(goal.id, "test_context", context)

        # Retrieve context
        retrieved = service._memory_manager.get_context(goal.id, "test_context")
        assert retrieved == context

    def test_error_handling_in_background_processing(self):
        """Test error handling in background processing loop."""
        service = GoalithService()

        # Mock the update processor to raise an error
        with patch.object(service._update_processor, "process_update") as mock_process:
            mock_process.side_effect = Exception("Processing error")

            # Start background processing
            service.start_background_processing()

            # Queue an update
            update = UpdateEvent(update_type=UpdateType.STATUS_CHANGE, node_id="test", data={})
            service.queue_update(update)

            # Give background thread time to process
            time.sleep(0.1)

            # Background thread should still be alive despite the error
            assert service._background_thread.is_alive()

            # Clean up
            service.stop_background_processing()

    def test_bulk_operations(self):
        """Test bulk operations for performance."""
        service = GoalithService()

        # Add multiple goals
        goal_ids = []
        for i in range(10):
            goal = service.add_goal(f"Goal {i}", priority=float(i))
            goal_ids.append(goal.id)

        # Verify all goals were added
        all_goals = service.list_goals()
        assert len(all_goals) == 10

        # Update multiple goals
        for goal_id in goal_ids[:5]:
            service.update_goal(goal_id, status=NodeStatus.COMPLETED)

        # Verify status updates
        completed_goals = [g for g in service.list_goals() if g.status == NodeStatus.COMPLETED]
        assert len(completed_goals) == 5

    def test_concurrent_access(self):
        """Test concurrent access to the service."""
        service = GoalithService()
        results = []
        errors = []

        def add_goals(start_idx, count):
            try:
                for i in range(start_idx, start_idx + count):
                    goal = service.add_goal(f"Concurrent Goal {i}")
                    results.append(goal.id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads adding goals concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_goals, args=(i * 10, 10))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all goals were added
        assert len(results) == 50
        assert len(set(results)) == 50  # All unique IDs

        # Verify goals are in the service
        all_goals = service.list_goals()
        assert len(all_goals) == 50
