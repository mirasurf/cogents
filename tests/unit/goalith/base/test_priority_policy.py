"""
Unit tests for goalith.base.priority_policy module.
"""
import pytest

from cogents.goalith.base.goal_node import GoalNode
from cogents.goalith.base.priority_policy import PriorityOrder, PriorityPolicy


class TestPriorityOrder:
    """Test PriorityOrder enum."""

    def test_priority_order_values(self):
        """Test that PriorityOrder has correct values."""
        assert PriorityOrder.HIGHEST_FIRST == "highest_first"
        assert PriorityOrder.LOWEST_FIRST == "lowest_first"


class ConcretePriorityPolicy(PriorityPolicy):
    """Concrete implementation for testing abstract base class."""

    def compare(self, node1, node2):
        """Compare two nodes by priority."""
        if node1.priority > node2.priority:
            return -1
        elif node1.priority < node2.priority:
            return 1
        else:
            return 0

    def score(self, node):
        """Return the node's priority as its score."""
        return node.priority

    @property
    def name(self):
        """Return policy name."""
        return "TestPriorityPolicy"


class TestPriorityPolicy:
    """Test PriorityPolicy abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that PriorityPolicy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PriorityPolicy()

    def test_concrete_implementation_compare_method(self):
        """Test that concrete implementation compare method works."""
        policy = ConcretePriorityPolicy()

        node_high = GoalNode(description="High", priority=9.0)
        node_low = GoalNode(description="Low", priority=1.0)
        node_equal = GoalNode(description="Equal", priority=9.0)

        # High priority node should compare as higher
        assert policy.compare(node_high, node_low) == -1
        assert policy.compare(node_low, node_high) == 1

        # Equal priority nodes should compare as equal
        assert policy.compare(node_high, node_equal) == 0

    def test_concrete_implementation_score_method(self):
        """Test that concrete implementation score method works."""
        policy = ConcretePriorityPolicy()

        node = GoalNode(description="Test", priority=7.5)
        score = policy.score(node)

        assert score == 7.5

    def test_concrete_implementation_name_property(self):
        """Test that concrete implementation name property works."""
        policy = ConcretePriorityPolicy()

        assert policy.name == "TestPriorityPolicy"

    def test_sort_nodes_method(self):
        """Test the default sort_nodes implementation."""
        policy = ConcretePriorityPolicy()

        nodes = [
            GoalNode(description="Low", priority=1.0),
            GoalNode(description="High", priority=9.0),
            GoalNode(description="Medium", priority=5.0),
        ]

        sorted_nodes = policy.sort_nodes(nodes)
        priorities = [n.priority for n in sorted_nodes]

        # Should be sorted highest priority first
        assert priorities == [9.0, 5.0, 1.0]

    def test_sort_nodes_empty_list(self):
        """Test sort_nodes with empty list."""
        policy = ConcretePriorityPolicy()

        result = policy.sort_nodes([])
        assert result == []

    def test_sort_nodes_single_node(self):
        """Test sort_nodes with single node."""
        policy = ConcretePriorityPolicy()

        node = GoalNode(description="Single", priority=7.0)
        result = policy.sort_nodes([node])

        assert len(result) == 1
        assert result[0] is node

    def test_sort_nodes_preserves_equal_priorities(self):
        """Test that sort_nodes handles equal priorities correctly."""
        policy = ConcretePriorityPolicy()

        nodes = [
            GoalNode(description="First", priority=5.0),
            GoalNode(description="Second", priority=5.0),
            GoalNode(description="Third", priority=5.0),
        ]

        sorted_nodes = policy.sort_nodes(nodes)

        # All should have same priority
        assert len(sorted_nodes) == 3
        assert all(n.priority == 5.0 for n in sorted_nodes)
