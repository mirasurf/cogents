"""
Shared fixtures for goalith tests.
"""
import pytest

# Import only the base classes that don't trigger external dependencies
from cogents.goalith.base.goal_node import GoalNode, NodeStatus, NodeType
from cogents.goalith.base.graph_store import GraphStore
from cogents.goalith.base.update_event import UpdateEvent, UpdateType
from cogents.goalith.memory.inmemstore import InMemoryStore

# Avoid importing decomposer modules that trigger LLM client initialization
# We'll create mock fixtures instead


@pytest.fixture
def sample_goal_node():
    """Create a sample goal node for testing."""
    return GoalNode(
        id="test-goal-1",
        description="Test goal for unit tests",
        type=NodeType.GOAL,
        status=NodeStatus.PENDING,
        priority=5.0,
        tags=["test", "sample"],
        context={"test": True},
    )


@pytest.fixture
def sample_task_node():
    """Create a sample task node for testing."""
    return GoalNode(
        id="test-task-1",
        description="Test task for unit tests",
        type=NodeType.TASK,
        status=NodeStatus.PENDING,
        priority=3.0,
        tags=["test", "task"],
        context={"complexity": "low"},
    )


@pytest.fixture
def sample_subgoal_node():
    """Create a sample subgoal node for testing."""
    return GoalNode(
        id="test-subgoal-1",
        description="Test subgoal for unit tests",
        type=NodeType.SUBGOAL,
        status=NodeStatus.IN_PROGRESS,
        priority=4.0,
        dependencies={"test-task-1"},
        tags=["test", "subgoal"],
    )


@pytest.fixture
def empty_graph_store():
    """Create an empty graph store for testing."""
    return GraphStore()


@pytest.fixture
def populated_graph_store(sample_goal_node, sample_subgoal_node, sample_task_node):
    """Create a graph store with sample nodes for testing."""
    store = GraphStore()
    store.add_node(sample_goal_node)
    store.add_node(sample_subgoal_node)
    store.add_node(sample_task_node)

    # Add some dependencies
    store.add_dependency(sample_subgoal_node.id, sample_task_node.id)
    store.add_dependency(sample_goal_node.id, sample_subgoal_node.id)

    return store


@pytest.fixture
def sample_update_event():
    """Create a sample update event for testing."""
    return UpdateEvent(
        id="test-update-1",
        update_type=UpdateType.STATUS_CHANGE,
        node_id="test-node-1",
        data={"old_status": "pending", "new_status": "in_progress"},
        source="test",
    )


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryStore()


@pytest.fixture
def decomposer_registry():
    """Create a mock decomposer registry for testing."""

    # Create a mock registry without importing the actual decomposer modules
    class MockDecomposerRegistry:
        def __init__(self):
            self._decomposers = {}

        def register(self, name, decomposer):
            self._decomposers[name] = decomposer

        def get(self, name):
            return self._decomposers.get(name)

        def list(self):
            return list(self._decomposers.keys())

    registry = MockDecomposerRegistry()
    # Register mock decomposers instead of real ones
    registry.register("simple", "mock_simple_decomposer")
    registry.register("human", "mock_human_decomposer")
    return registry


@pytest.fixture
def mock_context():
    """Create mock context data for testing."""
    return {
        "domain": "test",
        "user_preferences": {"priority_style": "deadline"},
        "historical_patterns": [],
        "current_workload": 3,
    }
