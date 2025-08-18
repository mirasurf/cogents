"""
Unit tests for goalith.base.goal_node module.
"""
from datetime import datetime, timezone
from uuid import UUID

import pytest

from cogents.goalith.base.goal_node import GoalNode, NodeStatus, NodeType


class TestNodeType:
    """Test NodeType enum."""

    def test_node_type_values(self):
        """Test that NodeType has correct values."""
        assert NodeType.GOAL == "goal"
        assert NodeType.SUBGOAL == "subgoal"
        assert NodeType.TASK == "task"

    def test_node_type_string_representation(self):
        """Test string representation of NodeType."""
        # Pydantic enums use value for string representation
        assert NodeType.GOAL.value == "goal"
        assert NodeType.SUBGOAL.value == "subgoal"
        assert NodeType.TASK.value == "task"


class TestNodeStatus:
    """Test NodeStatus enum."""

    def test_node_status_values(self):
        """Test that NodeStatus has correct values."""
        assert NodeStatus.PENDING == "pending"
        assert NodeStatus.IN_PROGRESS == "in_progress"
        assert NodeStatus.COMPLETED == "completed"
        assert NodeStatus.FAILED == "failed"
        assert NodeStatus.CANCELLED == "cancelled"
        assert NodeStatus.BLOCKED == "blocked"

    def test_node_status_string_representation(self):
        """Test string representation of NodeStatus."""
        # Pydantic enums use value for string representation
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.IN_PROGRESS.value == "in_progress"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.CANCELLED.value == "cancelled"
        assert NodeStatus.BLOCKED.value == "blocked"


class TestGoalNode:
    """Test GoalNode data model."""

    def test_default_creation(self):
        """Test creating a GoalNode with minimal parameters."""
        node = GoalNode(description="Test goal")
        
        # Check that ID is generated
        assert node.id is not None
        assert isinstance(UUID(node.id), UUID)  # Valid UUID format
        
        # Check defaults
        assert node.description == "Test goal"
        assert node.type == NodeType.TASK  # Default type
        assert node.status == NodeStatus.PENDING  # Default status
        assert node.priority == 1.0  # Default priority
        assert node.dependencies == set()
        assert node.children == set()
        assert node.parent is None
        assert node.tags == []
        assert node.context == {}
        assert node.notes is None
        assert node.deadline is None
        assert node.estimated_effort is None
        assert node.decomposer_name is None
        assert node.assigned_to is None
        assert node.created_at is not None
        assert node.updated_at is not None
        assert node.started_at is None
        assert node.completed_at is None
        assert node.retry_count == 0
        assert node.max_retries == 3
        assert node.error_message is None
        assert node.execution_notes == []
        assert node.performance_metrics == {}

    def test_custom_creation(self):
        """Test creating a GoalNode with custom parameters."""
        custom_id = "custom-goal-123"
        created_at = datetime.now(timezone.utc)
        deadline = datetime.now(timezone.utc)

        node = GoalNode(
            id=custom_id,
            description="Custom test goal",
            type=NodeType.GOAL,
            status=NodeStatus.IN_PROGRESS,
            priority=7.5,
            dependencies={"dep1", "dep2"},
            children={"child1"},
            parent="parent1",
            tags=["work", "important"],
            context={"project": "test", "version": 1},
            notes="Some notes",
            deadline=deadline,
            estimated_effort="2 hours",
            decomposer_name="test_decomposer",
            assigned_to="user123",
            created_at=created_at,
        )
        
        assert node.id == custom_id
        assert node.description == "Custom test goal"
        assert node.type == NodeType.GOAL
        assert node.status == NodeStatus.IN_PROGRESS
        assert node.priority == 7.5
        assert node.dependencies == {"dep1", "dep2"}
        assert node.children == {"child1"}
        assert node.parent == "parent1"
        assert node.tags == ["work", "important"]
        assert node.context == {"project": "test", "version": 1}
        assert node.notes == "Some notes"
        assert node.deadline == deadline
        assert node.estimated_effort == "2 hours"
        assert node.decomposer_name == "test_decomposer"
        assert node.assigned_to == "user123"
        assert node.created_at == created_at

    def test_is_ready_pending_status(self):
        """Test is_ready with pending status."""
        node = GoalNode(description="Test", status=NodeStatus.PENDING)
        assert node.is_ready() is True

    def test_is_ready_non_pending_status(self):
        """Test is_ready with non-pending status."""
        node = GoalNode(description="Test", status=NodeStatus.COMPLETED)
        assert node.is_ready() is False
        
        node.status = NodeStatus.IN_PROGRESS
        assert node.is_ready() is False
        
        node.status = NodeStatus.FAILED
        assert node.is_ready() is False
        
        node.status = NodeStatus.CANCELLED
        assert node.is_ready() is False
        
        node.status = NodeStatus.BLOCKED
        assert node.is_ready() is False

    def test_is_terminal_status(self):
        """Test is_terminal method."""
        node = GoalNode(description="Test")

        # Non-terminal statuses
        node.status = NodeStatus.PENDING
        assert node.is_terminal() is False

        node.status = NodeStatus.IN_PROGRESS
        assert node.is_terminal() is False

        node.status = NodeStatus.BLOCKED
        assert node.is_terminal() is False

        # Terminal statuses
        node.status = NodeStatus.COMPLETED
        assert node.is_terminal() is True

        node.status = NodeStatus.FAILED
        assert node.is_terminal() is True

        node.status = NodeStatus.CANCELLED
        assert node.is_terminal() is True

    def test_status_update_methods(self):
        """Test status update methods."""
        node = GoalNode(description="Test", status=NodeStatus.PENDING)
        old_updated_at = node.updated_at
        
        # Mark started (to in_progress)
        node.mark_started()
        assert node.status == NodeStatus.IN_PROGRESS
        assert node.started_at is not None
        assert node.updated_at > old_updated_at
        
        # Mark completed
        old_updated_at = node.updated_at
        node.mark_completed()
        assert node.status == NodeStatus.COMPLETED
        assert node.completed_at is not None
        assert node.updated_at > old_updated_at

    def test_mark_failed_method(self):
        """Test mark_failed method."""
        node = GoalNode(description="Test", status=NodeStatus.IN_PROGRESS)
        old_retry_count = node.retry_count
        
        node.mark_failed("Test error")
        assert node.status == NodeStatus.FAILED
        assert node.error_message == "Test error"
        assert node.retry_count == old_retry_count + 1

    def test_mark_cancelled_method(self):
        """Test mark_cancelled method."""
        node = GoalNode(description="Test", status=NodeStatus.IN_PROGRESS)
        
        node.mark_cancelled()
        assert node.status == NodeStatus.CANCELLED

    def test_equality(self):
        """Test GoalNode equality."""
        node1 = GoalNode(id="test-1", description="Test")
        node2 = GoalNode(id="test-1", description="Test")
        node3 = GoalNode(id="test-2", description="Test")

        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert node1 != "not-a-node"  # Different type

    def test_model_serialization(self):
        """Test model serialization using Pydantic methods."""
        node = GoalNode(id="test-1", description="Test goal", type=NodeType.GOAL, priority=5.0, tags=["tag1", "tag2"])

        result = node.model_dump()

        assert isinstance(result, dict)
        assert result["id"] == "test-1"
        assert result["description"] == "Test goal"
        assert result["type"] == "goal"
        assert result["priority"] == 5.0
        assert "created_at" in result
        assert "updated_at" in result

    def test_model_creation_from_dict(self):
        """Test model creation from dict using Pydantic methods."""
        data = {
            "id": "test-1",
            "description": "Test goal",
            "type": "goal",
            "status": "pending",
            "priority": 5.0,
            "dependencies": ["dep1", "dep2"],
            "tags": ["tag1", "tag2"],
            "context": {"key": "value"},
            "created_at": "2023-01-01T00:00:00+00:00",
            "updated_at": "2023-01-01T00:00:00+00:00",
        }

        node = GoalNode.model_validate(data)

        assert node.id == "test-1"
        assert node.description == "Test goal"
        assert node.type == NodeType.GOAL
        assert node.status == NodeStatus.PENDING
        assert node.priority == 5.0
        assert node.dependencies == {"dep1", "dep2"}
        assert node.tags == ["tag1", "tag2"]
        assert node.context == {"key": "value"}

    def test_validate_priority_range(self):
        """Test that priority is properly validated."""
        # Valid priorities
        node = GoalNode(description="Test", priority=0.0)
        assert node.priority == 0.0

        node = GoalNode(description="Test", priority=10.0)
        assert node.priority == 10.0

        node = GoalNode(description="Test", priority=5.5)
        assert node.priority == 5.5

    def test_description_required(self):
        """Test that description is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            GoalNode()

    def test_tags_and_dependencies_handling(self):
        """Test that tags and dependencies are properly handled."""
        node = GoalNode(
            description="Test",
            dependencies={"dep1", "dep2"},  # Set
            tags=["tag1", "tag2", "tag1"]  # List with duplicate
        )
        
        # Dependencies are sets, tags are lists
        assert node.dependencies == {"dep1", "dep2"}
        assert node.tags == ["tag1", "tag2", "tag1"]  # Duplicates preserved in list

    def test_add_remove_dependencies(self):
        """Test add/remove dependency methods."""
        node = GoalNode(description="Test")
        
        node.add_dependency("dep1")
        assert "dep1" in node.dependencies
        
        node.remove_dependency("dep1")
        assert "dep1" not in node.dependencies

    def test_add_remove_children(self):
        """Test add/remove child methods."""
        node = GoalNode(description="Test")
        
        node.add_child("child1")
        assert "child1" in node.children
        
        node.remove_child("child1")
        assert "child1" not in node.children

    def test_add_note_method(self):
        """Test add_note method."""
        node = GoalNode(description="Test")
        
        node.add_note("Test note")
        assert len(node.execution_notes) == 1
        assert "Test note" in node.execution_notes[0]

    def test_update_context_method(self):
        """Test update_context method."""
        node = GoalNode(description="Test")
        
        node.update_context("key1", "value1")
        assert node.context["key1"] == "value1"

    def test_can_retry_method(self):
        """Test can_retry method."""
        node = GoalNode(description="Test", status=NodeStatus.FAILED, retry_count=1, max_retries=3)
        assert node.can_retry() is True
        
        node.retry_count = 3
        assert node.can_retry() is False
        
        node.status = NodeStatus.COMPLETED
        assert node.can_retry() is False
