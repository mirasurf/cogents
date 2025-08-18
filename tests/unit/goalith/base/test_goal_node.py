"""
Unit tests for goalith.base.goal_node module.
"""
import pytest
from datetime import datetime, timezone
from uuid import UUID

from cogents.goalith.base.goal_node import GoalNode, NodeType, NodeStatus


class TestNodeType:
    """Test NodeType enum."""

    def test_node_type_values(self):
        """Test that NodeType has correct values."""
        assert NodeType.GOAL == "goal"
        assert NodeType.SUBGOAL == "subgoal"
        assert NodeType.TASK == "task"

    def test_node_type_string_representation(self):
        """Test string representation of NodeType."""
        assert str(NodeType.GOAL) == "goal"
        assert str(NodeType.SUBGOAL) == "subgoal"
        assert str(NodeType.TASK) == "task"


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
        assert str(NodeStatus.PENDING) == "pending"
        assert str(NodeStatus.IN_PROGRESS) == "in_progress"
        assert str(NodeStatus.COMPLETED) == "completed"
        assert str(NodeStatus.FAILED) == "failed"
        assert str(NodeStatus.CANCELLED) == "cancelled"
        assert str(NodeStatus.BLOCKED) == "blocked"


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
        assert node.tags == set()
        assert node.metadata == {}
        assert node.notes is None
        assert node.deadline is None
        assert node.estimated_effort is None
        assert node.actual_effort is None
        assert node.created_at is not None
        assert node.updated_at is not None
        assert node.started_at is None
        assert node.completed_at is None

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
            tags={"work", "important"},
            metadata={"project": "test", "version": 1},
            notes="Some notes",
            deadline=deadline,
            estimated_effort="2 hours",
            created_at=created_at
        )
        
        assert node.id == custom_id
        assert node.description == "Custom test goal"
        assert node.type == NodeType.GOAL
        assert node.status == NodeStatus.IN_PROGRESS
        assert node.priority == 7.5
        assert node.dependencies == {"dep1", "dep2"}
        assert node.tags == {"work", "important"}
        assert node.metadata == {"project": "test", "version": 1}
        assert node.notes == "Some notes"
        assert node.deadline == deadline
        assert node.estimated_effort == "2 hours"
        assert node.created_at == created_at

    def test_is_ready_no_dependencies(self):
        """Test is_ready with no dependencies."""
        node = GoalNode(description="Test", status=NodeStatus.PENDING)
        assert node.is_ready(set()) is True
        assert node.is_ready({"other-node"}) is True

    def test_is_ready_with_dependencies_satisfied(self):
        """Test is_ready with satisfied dependencies."""
        node = GoalNode(
            description="Test",
            status=NodeStatus.PENDING,
            dependencies={"dep1", "dep2"}
        )
        completed_nodes = {"dep1", "dep2", "extra"}
        assert node.is_ready(completed_nodes) is True

    def test_is_ready_with_dependencies_not_satisfied(self):
        """Test is_ready with unsatisfied dependencies."""
        node = GoalNode(
            description="Test",
            status=NodeStatus.PENDING,
            dependencies={"dep1", "dep2"}
        )
        completed_nodes = {"dep1"}  # Missing dep2
        assert node.is_ready(completed_nodes) is False

    def test_is_ready_non_pending_status(self):
        """Test is_ready with non-pending status."""
        node = GoalNode(description="Test", status=NodeStatus.COMPLETED)
        assert node.is_ready(set()) is False
        
        node.status = NodeStatus.IN_PROGRESS
        assert node.is_ready(set()) is False
        
        node.status = NodeStatus.FAILED
        assert node.is_ready(set()) is False

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

    def test_update_status(self):
        """Test update_status method."""
        node = GoalNode(description="Test", status=NodeStatus.PENDING)
        old_updated_at = node.updated_at
        
        # Update to in_progress
        node.update_status(NodeStatus.IN_PROGRESS)
        assert node.status == NodeStatus.IN_PROGRESS
        assert node.started_at is not None
        assert node.updated_at > old_updated_at
        
        # Update to completed
        old_updated_at = node.updated_at
        node.update_status(NodeStatus.COMPLETED)
        assert node.status == NodeStatus.COMPLETED
        assert node.completed_at is not None
        assert node.updated_at > old_updated_at

    def test_equality(self):
        """Test GoalNode equality."""
        node1 = GoalNode(id="test-1", description="Test")
        node2 = GoalNode(id="test-1", description="Test")
        node3 = GoalNode(id="test-2", description="Test")
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert node1 != "not-a-node"  # Different type

    def test_to_dict(self):
        """Test to_dict method."""
        node = GoalNode(
            id="test-1",
            description="Test goal",
            type=NodeType.GOAL,
            priority=5.0,
            tags={"tag1", "tag2"}
        )
        
        result = node.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "test-1"
        assert result["description"] == "Test goal"
        assert result["type"] == "goal"
        assert result["priority"] == 5.0
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "id": "test-1",
            "description": "Test goal",
            "type": "goal",
            "status": "pending",
            "priority": 5.0,
            "dependencies": ["dep1", "dep2"],
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "value"},
            "created_at": "2023-01-01T00:00:00+00:00",
            "updated_at": "2023-01-01T00:00:00+00:00"
        }
        
        node = GoalNode.from_dict(data)
        
        assert node.id == "test-1"
        assert node.description == "Test goal"
        assert node.type == NodeType.GOAL
        assert node.status == NodeStatus.PENDING
        assert node.priority == 5.0
        assert node.dependencies == {"dep1", "dep2"}
        assert node.tags == {"tag1", "tag2"}
        assert node.metadata == {"key": "value"}

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

    def test_tags_and_dependencies_are_sets(self):
        """Test that tags and dependencies are properly handled as sets."""
        node = GoalNode(
            description="Test",
            dependencies=["dep1", "dep2", "dep1"],  # Duplicate
            tags=["tag1", "tag2", "tag1"]  # Duplicate
        )
        
        # Sets should remove duplicates
        assert node.dependencies == {"dep1", "dep2"}
        assert node.tags == {"tag1", "tag2"}