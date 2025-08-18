"""
Unit tests for goalith.schedule.scheduler module.
"""
from datetime import datetime, timedelta, timezone

from cogents.goalith.base.goal_node import GoalNode, NodeStatus
from cogents.goalith.schedule.deadline_priority_policy import DeadlinePriorityPolicy
from cogents.goalith.schedule.scheduler import Scheduler
from cogents.goalith.schedule.simple_priority_policy import SimplePriorityPolicy


class TestScheduler:
    """Test Scheduler functionality."""

    def test_default_initialization(self):
        """Test creating scheduler with default policy."""
        scheduler = Scheduler()

        assert scheduler._policy is not None
        assert isinstance(scheduler._policy, SimplePriorityPolicy)
        assert scheduler._stats["schedule_calls"] == 0
        assert scheduler._stats["nodes_scheduled"] == 0
        assert scheduler._stats["get_next_calls"] == 0

    def test_custom_policy_initialization(self):
        """Test creating scheduler with custom policy."""
        custom_policy = DeadlinePriorityPolicy()
        scheduler = Scheduler(priority_policy=custom_policy)

        assert scheduler._policy is custom_policy

    def test_backward_compatibility_policy_parameter(self):
        """Test backward compatibility with 'policy' parameter."""
        custom_policy = SimplePriorityPolicy()
        scheduler = Scheduler(policy=custom_policy)

        assert scheduler._policy is custom_policy

    def test_priority_policy_parameter_takes_precedence(self):
        """Test that priority_policy parameter takes precedence over policy."""
        policy1 = SimplePriorityPolicy()
        policy2 = DeadlinePriorityPolicy()

        scheduler = Scheduler(priority_policy=policy1, policy=policy2)

        assert scheduler._policy is policy1

    def test_set_policy(self):
        """Test setting a new policy."""
        scheduler = Scheduler()
        original_policy = scheduler._policy

        new_policy = DeadlinePriorityPolicy()
        scheduler.set_policy(new_policy)

        assert scheduler._policy is new_policy
        assert scheduler._policy is not original_policy

    def test_get_policy(self):
        """Test getting the current policy."""
        policy = SimplePriorityPolicy()
        scheduler = Scheduler(priority_policy=policy)

        retrieved_policy = scheduler.get_policy()
        assert retrieved_policy is policy

    def test_schedule_empty_list(self):
        """Test scheduling empty list of nodes."""
        scheduler = Scheduler()

        result = scheduler.schedule([])

        assert result == []
        assert scheduler._stats["schedule_calls"] == 1
        assert scheduler._stats["nodes_scheduled"] == 0

    def test_schedule_single_node(self):
        """Test scheduling single node."""
        scheduler = Scheduler()
        node = GoalNode(description="Test node", priority=5.0)

        result = scheduler.schedule([node])

        assert len(result) == 1
        assert result[0] is node
        assert scheduler._stats["schedule_calls"] == 1
        assert scheduler._stats["nodes_scheduled"] == 1

    def test_schedule_multiple_nodes_by_priority(self):
        """Test scheduling multiple nodes by priority."""
        scheduler = Scheduler()

        low_priority = GoalNode(description="Low", priority=1.0)
        high_priority = GoalNode(description="High", priority=9.0)
        medium_priority = GoalNode(description="Medium", priority=5.0)

        nodes = [low_priority, high_priority, medium_priority]
        result = scheduler.schedule(nodes)

        # Should be ordered by priority (highest first)
        assert len(result) == 3
        assert result[0] is high_priority
        assert result[1] is medium_priority
        assert result[2] is low_priority

        assert scheduler._stats["nodes_scheduled"] == 3

    def test_schedule_with_deadline_policy(self):
        """Test scheduling with deadline-based policy."""
        deadline_policy = DeadlinePriorityPolicy()
        scheduler = Scheduler(priority_policy=deadline_policy)

        now = datetime.now(timezone.utc)
        soon_deadline = now + timedelta(hours=1)
        late_deadline = now + timedelta(days=1)

        soon_node = GoalNode(description="Soon", deadline=soon_deadline, priority=1.0)
        late_node = GoalNode(description="Late", deadline=late_deadline, priority=9.0)

        result = scheduler.schedule([soon_node, late_node])

        # Soon deadline should come first despite lower priority
        assert result[0] is soon_node
        assert result[1] is late_node

    def test_get_next_empty_list(self):
        """Test getting next node from empty list."""
        scheduler = Scheduler()

        result = scheduler.get_next([])

        assert result is None
        assert scheduler._stats["get_next_calls"] == 1

    def test_get_next_single_node(self):
        """Test getting next node from single node list."""
        scheduler = Scheduler()
        node = GoalNode(description="Only node", priority=5.0)

        result = scheduler.get_next([node])

        assert result is node
        assert scheduler._stats["get_next_calls"] == 1

    def test_get_next_multiple_nodes(self):
        """Test getting highest priority node from multiple nodes."""
        scheduler = Scheduler()

        low_priority = GoalNode(description="Low", priority=1.0)
        high_priority = GoalNode(description="High", priority=9.0)
        medium_priority = GoalNode(description="Medium", priority=5.0)

        nodes = [low_priority, high_priority, medium_priority]
        result = scheduler.get_next(nodes)

        assert result is high_priority

    def test_get_next_with_equal_priorities(self):
        """Test getting next node when priorities are equal."""
        scheduler = Scheduler()

        node1 = GoalNode(id="1", description="Node 1", priority=5.0)
        node2 = GoalNode(id="2", description="Node 2", priority=5.0)
        node3 = GoalNode(id="3", description="Node 3", priority=5.0)

        nodes = [node1, node2, node3]
        result = scheduler.get_next(nodes)

        # Should return one of them (deterministic based on policy implementation)
        assert result in nodes

    def test_peek_all_empty_list(self):
        """Test peeking at all nodes in empty list."""
        scheduler = Scheduler()

        result = scheduler.peek_all([])

        assert result == []

    def test_peek_all_preserves_input(self):
        """Test that peek_all doesn't modify input list."""
        scheduler = Scheduler()

        low_priority = GoalNode(description="Low", priority=1.0)
        high_priority = GoalNode(description="High", priority=9.0)

        original = [low_priority, high_priority]
        result = scheduler.peek_all(original)

        # Original list should be unchanged
        assert original == [low_priority, high_priority]

        # Result should be sorted
        assert result == [high_priority, low_priority]
        assert result is not original

    def test_peek_all_multiple_nodes(self):
        """Test peeking at all nodes in priority order."""
        scheduler = Scheduler()

        nodes = [
            GoalNode(description="Priority 3", priority=3.0),
            GoalNode(description="Priority 7", priority=7.0),
            GoalNode(description="Priority 1", priority=1.0),
            GoalNode(description="Priority 5", priority=5.0),
        ]

        result = scheduler.peek_all(nodes)

        # Should be in descending priority order
        expected_priorities = [7.0, 5.0, 3.0, 1.0]
        actual_priorities = [node.priority for node in result]
        assert actual_priorities == expected_priorities

    def test_get_stats(self):
        """Test getting scheduler statistics."""
        scheduler = Scheduler()

        # Perform some operations
        nodes = [GoalNode(description="Test", priority=5.0)]
        scheduler.schedule(nodes)
        scheduler.get_next(nodes)
        scheduler.peek_all(nodes)

        stats = scheduler.get_stats()

        assert stats["schedule_calls"] == 1
        assert stats["nodes_scheduled"] == 1
        assert stats["get_next_calls"] == 1

    def test_reset_stats(self):
        """Test resetting scheduler statistics."""
        scheduler = Scheduler()

        # Perform some operations
        nodes = [GoalNode(description="Test", priority=5.0)]
        scheduler.schedule(nodes)
        scheduler.get_next(nodes)

        # Verify stats are non-zero
        assert scheduler._stats["schedule_calls"] > 0
        assert scheduler._stats["get_next_calls"] > 0

        # Reset stats
        scheduler.reset_stats()

        # Verify stats are zero
        assert scheduler._stats["schedule_calls"] == 0
        assert scheduler._stats["nodes_scheduled"] == 0
        assert scheduler._stats["get_next_calls"] == 0

    def test_complex_scheduling_scenario(self):
        """Test complex scheduling scenario with mixed priorities and deadlines."""
        # Use simple priority policy
        scheduler = Scheduler(priority_policy=SimplePriorityPolicy())

        now = datetime.now(timezone.utc)

        nodes = [
            GoalNode(description="High priority urgent", priority=9.0, deadline=now + timedelta(hours=1)),
            GoalNode(description="Low priority not urgent", priority=2.0, deadline=now + timedelta(days=7)),
            GoalNode(description="Medium priority", priority=5.0),
            GoalNode(description="High priority not urgent", priority=8.0, deadline=now + timedelta(days=2)),
        ]

        # Schedule all nodes
        scheduled = scheduler.schedule(nodes)

        # With simple priority policy, should be ordered by priority
        priorities = [node.priority for node in scheduled]
        assert priorities == sorted(priorities, reverse=True)

        # Get next node
        next_node = scheduler.get_next(nodes)
        assert next_node.priority == 9.0

    def test_filter_ready_nodes(self):
        """Test that scheduler respects node readiness."""
        scheduler = Scheduler()

        # Create nodes with different statuses
        pending_node = GoalNode(description="Pending", priority=5.0, status=NodeStatus.PENDING)
        in_progress_node = GoalNode(description="In Progress", priority=7.0, status=NodeStatus.IN_PROGRESS)
        completed_node = GoalNode(description="Completed", priority=9.0, status=NodeStatus.COMPLETED)

        # Only pending nodes should be considered for scheduling in most contexts
        nodes = [pending_node, in_progress_node, completed_node]

        # Test with all nodes - scheduler should handle this appropriately
        result = scheduler.schedule(nodes)

        # All nodes should be returned in priority order
        # (The scheduler itself doesn't filter by status - that's the graph store's job)
        assert len(result) == 3
        priorities = [node.priority for node in result]
        assert priorities == sorted(priorities, reverse=True)

    def test_scheduling_performance_with_many_nodes(self):
        """Test scheduler performance with many nodes."""
        scheduler = Scheduler()

        # Create many nodes with random priorities
        import random

        nodes = []
        for i in range(1000):
            priority = random.uniform(0.0, 10.0)
            node = GoalNode(description=f"Node {i}", priority=priority)
            nodes.append(node)

        # Schedule all nodes
        result = scheduler.schedule(nodes)

        # Verify all nodes are present and properly sorted
        assert len(result) == 1000
        priorities = [node.priority for node in result]
        assert priorities == sorted(priorities, reverse=True)

        # Get next node should be highest priority
        next_node = scheduler.get_next(nodes)
        assert next_node.priority == max(node.priority for node in nodes)

    def test_peek_top_n(self):
        """Test getting top N nodes."""
        scheduler = Scheduler()

        nodes = [
            GoalNode(description="Priority 9", priority=9.0),
            GoalNode(description="Priority 8", priority=8.0),
            GoalNode(description="Priority 7", priority=7.0),
            GoalNode(description="Priority 6", priority=6.0),
        ]

        # Test getting top 2
        result = scheduler.peek_top_n(nodes, 2)
        assert len(result) == 2
        assert result[0].priority == 9.0
        assert result[1].priority == 8.0

        # Test getting more than available
        result = scheduler.peek_top_n(nodes, 10)
        assert len(result) == 4

    def test_schedule_with_limit(self):
        """Test scheduling with a limit on returned nodes."""
        scheduler = Scheduler()

        nodes = [
            GoalNode(description="Priority 9", priority=9.0),
            GoalNode(description="Priority 8", priority=8.0),
            GoalNode(description="Priority 7", priority=7.0),
            GoalNode(description="Priority 6", priority=6.0),
        ]

        # Test with limit
        result = scheduler.schedule(nodes, limit=2)
        assert len(result) == 2
        assert result[0].priority == 9.0
        assert result[1].priority == 8.0

        # Test without limit
        result = scheduler.schedule(nodes)
        assert len(result) == 4
