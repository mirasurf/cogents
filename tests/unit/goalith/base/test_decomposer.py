"""
Unit tests for goalith.base.decomposer module.
"""
import pytest

from cogents.goalith.base.decomposer import GoalDecomposer
from cogents.goalith.base.goal_node import GoalNode, NodeType


class ConcreteDecomposer(GoalDecomposer):
    """Concrete implementation for testing abstract base class."""

    def decompose(self, goal, context=None):
        """Simple decomposition that creates subgoals based on goal priority."""
        subgoals = []
        num_subgoals = min(int(goal.priority), 5)  # Max 5 subgoals

        for i in range(num_subgoals):
            subgoal = GoalNode(
                description=f"{goal.description} - Subgoal {i+1}",
                type=NodeType.SUBGOAL,
                priority=goal.priority / (i + 1),
            )
            subgoals.append(subgoal)

        return subgoals


class TestGoalDecomposer:
    """Test GoalDecomposer abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that GoalDecomposer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GoalDecomposer()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works correctly."""
        decomposer = ConcreteDecomposer()

        goal = GoalNode(description="Main goal", type=NodeType.GOAL, priority=3.0)

        subgoals = decomposer.decompose(goal)

        assert len(subgoals) == 3  # Based on priority
        assert all(sg.type == NodeType.SUBGOAL for sg in subgoals)
        assert all("Main goal" in sg.description for sg in subgoals)

    def test_decompose_with_context(self):
        """Test decompose with context parameter."""
        decomposer = ConcreteDecomposer()

        goal = GoalNode(description="Test goal", priority=2.0)
        context = {"domain": "testing", "style": "detailed"}

        # Should not raise error
        subgoals = decomposer.decompose(goal, context=context)
        assert len(subgoals) == 2

    def test_decompose_empty_result(self):
        """Test decompose can return empty list."""
        decomposer = ConcreteDecomposer()

        goal = GoalNode(description="No subgoals", priority=0.0)

        subgoals = decomposer.decompose(goal)
        assert subgoals == []

    def test_decompose_preserves_goal_information(self):
        """Test that decomposition preserves relevant goal information."""
        decomposer = ConcreteDecomposer()

        goal = GoalNode(
            description="Complex goal",
            type=NodeType.GOAL,
            priority=4.0,
            tags={"important", "urgent"},
            metadata={"project": "test"},
        )

        subgoals = decomposer.decompose(goal)

        assert len(subgoals) == 4
        for subgoal in subgoals:
            # Should reference original goal
            assert "Complex goal" in subgoal.description
            # Should have appropriate type
            assert subgoal.type == NodeType.SUBGOAL
            # Should have derived priority
            assert subgoal.priority <= goal.priority
