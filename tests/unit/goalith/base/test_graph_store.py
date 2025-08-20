"""
Unit tests for goalith.base.graph_store module.
"""
import tempfile
from pathlib import Path

import pytest

from cogents.goalith.base.errors import CycleDetectedError, NodeNotFoundError
from cogents.goalith.base.goal_node import GoalNode, NodeStatus
from cogents.goalith.base.graph_store import GraphStore


class TestGraphStore:
    """Test GraphStore functionality."""

    def test_empty_initialization(self):
        """Test creating an empty graph store."""
        store = GraphStore()
        assert len(store._nodes) == 0
        assert store._graph.number_of_nodes() == 0
        assert store._graph.number_of_edges() == 0

    def test_add_node(self, sample_goal_node):
        """Test adding a node to the graph."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        assert sample_goal_node.id in store._nodes
        assert store._nodes[sample_goal_node.id] == sample_goal_node
        assert sample_goal_node.id in store._graph
        assert store._graph.number_of_nodes() == 1

    def test_add_duplicate_node_raises_error(self, sample_goal_node):
        """Test that adding a duplicate node raises ValueError."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        # Try to add the same node again
        with pytest.raises(ValueError, match=f"Node {sample_goal_node.id} already exists"):
            store.add_node(sample_goal_node)

    def test_get_node(self, sample_goal_node):
        """Test retrieving a node by ID."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        retrieved = store.get_node(sample_goal_node.id)
        assert retrieved == sample_goal_node

    def test_get_nonexistent_node_raises_error(self):
        """Test that getting a nonexistent node raises NodeNotFoundError."""
        store = GraphStore()

        with pytest.raises(NodeNotFoundError, match="Node nonexistent not found"):
            store.get_node("nonexistent")

    def test_update_node(self, sample_goal_node):
        """Test updating a node in the graph."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        # Modify the node
        sample_goal_node.status = NodeStatus.IN_PROGRESS
        sample_goal_node.priority = 8.0

        store.update_node(sample_goal_node)

        retrieved = store.get_node(sample_goal_node.id)
        assert retrieved.status == NodeStatus.IN_PROGRESS
        assert retrieved.priority == 8.0

    def test_update_nonexistent_node_raises_error(self, sample_goal_node):
        """Test that updating a nonexistent node raises NodeNotFoundError."""
        store = GraphStore()

        with pytest.raises(NodeNotFoundError, match=f"Node {sample_goal_node.id} not found"):
            store.update_node(sample_goal_node)

    def test_remove_node(self, sample_goal_node):
        """Test removing a node from the graph."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        # Verify node is there
        assert sample_goal_node.id in store._nodes

        # Remove it
        store.remove_node(sample_goal_node.id)

        # Verify it's gone
        assert sample_goal_node.id not in store._nodes
        assert sample_goal_node.id not in store._graph

    def test_remove_nonexistent_node_raises_error(self):
        """Test that removing a nonexistent node raises NodeNotFoundError."""
        store = GraphStore()

        with pytest.raises(NodeNotFoundError, match="Node nonexistent not found"):
            store.remove_node("nonexistent")

    def test_add_dependency(self, sample_goal_node, sample_task_node):
        """Test adding a dependency between nodes."""
        store = GraphStore()
        store.add_node(sample_goal_node)
        store.add_node(sample_task_node)

        store.add_dependency(sample_goal_node.id, sample_task_node.id)

        # Check graph structure
        assert store._graph.has_edge(sample_task_node.id, sample_goal_node.id)

        # Check node dependencies are updated
        assert sample_task_node.id in sample_goal_node.dependencies

    def test_add_dependency_nonexistent_nodes(self, sample_goal_node):
        """Test adding dependency with nonexistent nodes."""
        store = GraphStore()
        store.add_node(sample_goal_node)

        # Parent exists, child doesn't
        with pytest.raises(NodeNotFoundError):
            store.add_dependency(sample_goal_node.id, "nonexistent")

        # Child exists, parent doesn't
        with pytest.raises(NodeNotFoundError):
            store.add_dependency("nonexistent", sample_goal_node.id)

    def test_add_dependency_creates_cycle_raises_error(self):
        """Test that adding a dependency that creates a cycle raises CycleDetectedError."""
        store = GraphStore()

        # Create nodes: A -> B -> C
        node_a = GoalNode(id="a", description="Node A")
        node_b = GoalNode(id="b", description="Node B")
        node_c = GoalNode(id="c", description="Node C")

        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        store.add_dependency("a", "b")  # A depends on B
        store.add_dependency("b", "c")  # B depends on C

        # Try to create cycle: C -> A (which would create A -> B -> C -> A)
        with pytest.raises(CycleDetectedError):
            store.add_dependency("c", "a")

    def test_remove_dependency(self, sample_goal_node, sample_task_node):
        """Test removing a dependency between nodes."""
        store = GraphStore()
        store.add_node(sample_goal_node)
        store.add_node(sample_task_node)
        store.add_dependency(sample_goal_node.id, sample_task_node.id)

        # Verify dependency exists
        assert sample_task_node.id in sample_goal_node.dependencies

        # Remove dependency
        store.remove_dependency(sample_goal_node.id, sample_task_node.id)

        # Verify dependency is gone
        assert sample_task_node.id not in sample_goal_node.dependencies
        assert not store._graph.has_edge(sample_task_node.id, sample_goal_node.id)

    def test_remove_nonexistent_dependency(self, sample_goal_node, sample_task_node):
        """Test removing a nonexistent dependency."""
        store = GraphStore()
        store.add_node(sample_goal_node)
        store.add_node(sample_task_node)

        # Should not raise error, just do nothing
        store.remove_dependency(sample_goal_node.id, sample_task_node.id)

        # Verify no dependency was created
        assert sample_task_node.id not in sample_goal_node.dependencies

    def test_get_ready_nodes_no_dependencies(self):
        """Test getting ready nodes when no dependencies exist."""
        store = GraphStore()

        # Add some pending nodes
        node1 = GoalNode(id="1", description="Node 1", status=NodeStatus.PENDING)
        node2 = GoalNode(id="2", description="Node 2", status=NodeStatus.PENDING)
        node3 = GoalNode(id="3", description="Node 3", status=NodeStatus.COMPLETED)

        store.add_node(node1)
        store.add_node(node2)
        store.add_node(node3)

        ready_nodes = store.get_ready_nodes()

        # Only pending nodes should be ready
        ready_ids = {node.id for node in ready_nodes}
        assert ready_ids == {"1", "2"}

    def test_get_ready_nodes_with_dependencies(self):
        """Test getting ready nodes with dependency constraints."""
        store = GraphStore()

        # Create dependency chain: node1 -> node2 -> node3
        node1 = GoalNode(id="1", description="Node 1", status=NodeStatus.PENDING)
        node2 = GoalNode(id="2", description="Node 2", status=NodeStatus.PENDING)
        node3 = GoalNode(id="3", description="Node 3", status=NodeStatus.COMPLETED)
        node4 = GoalNode(id="4", description="Node 4", status=NodeStatus.PENDING)  # Independent

        store.add_node(node1)
        store.add_node(node2)
        store.add_node(node3)
        store.add_node(node4)

        store.add_dependency("1", "2")  # 1 depends on 2
        store.add_dependency("2", "3")  # 2 depends on 3

        ready_nodes = store.get_ready_nodes()
        ready_ids = {node.id for node in ready_nodes}

        # Only node2 (depends on completed node3) and node4 (independent) should be ready
        # node1 depends on node2, so it's not ready
        assert ready_ids == {"2", "4"}

    def test_get_children(self, populated_graph_store):
        """Test getting children of a node."""
        store = populated_graph_store

        # Get children of the subgoal node (it should have the goal as a child)
        children = store.get_children("test-subgoal-1")
        child_ids = {node.id for node in children}

        assert "test-goal-1" in child_ids

    def test_get_parents(self, populated_graph_store):
        """Test getting parents of a node."""
        store = populated_graph_store

        # Get parents of the subgoal node (it should have the task as a parent)
        parents = store.get_parents("test-subgoal-1")
        parent_ids = {node.id for node in parents}

        assert "test-task-1" in parent_ids

    def test_get_descendants(self, populated_graph_store):
        """Test getting all descendants of a node."""
        store = populated_graph_store

        # Get all descendants of the task node (it should have subgoal and goal as descendants)
        descendants = store.get_descendants("test-task-1")
        descendant_ids = set(descendants)  # These are already strings

        # Should include both subgoal and goal
        assert "test-subgoal-1" in descendant_ids
        assert "test-goal-1" in descendant_ids

    def test_get_ancestors(self, populated_graph_store):
        """Test getting all ancestors of a node."""
        store = populated_graph_store

        # Get all ancestors of the goal node (it should have subgoal and task as ancestors)
        ancestors = store.get_ancestors("test-goal-1")
        ancestor_ids = set(ancestors)  # These are already strings

        # Should include both subgoal and task
        assert "test-subgoal-1" in ancestor_ids
        assert "test-task-1" in ancestor_ids

    def test_list_nodes(self, populated_graph_store):
        """Test listing all nodes."""
        store = populated_graph_store

        all_nodes = store.list_nodes()
        node_ids = {node.id for node in all_nodes}

        expected_ids = {"test-goal-1", "test-subgoal-1", "test-task-1"}
        assert node_ids == expected_ids

    def test_has_node(self, populated_graph_store):
        """Test checking if node exists."""
        store = populated_graph_store

        assert store.has_node("test-goal-1") is True
        assert store.has_node("nonexistent") is False

    def test_get_node_count(self, populated_graph_store):
        """Test getting node count."""
        store = populated_graph_store
        assert store.get_node_count() == 3

    def test_save_and_load_graph(self, populated_graph_store):
        """Test saving and loading graph to/from file."""
        store = populated_graph_store

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save the graph
            store.save_graph(temp_path)

            # Create new store and load
            new_store = GraphStore()
            new_store.load_graph(temp_path)

            # Verify all nodes are loaded
            assert new_store.get_node_count() == 3
            assert new_store.has_node("test-goal-1")
            assert new_store.has_node("test-subgoal-1")
            assert new_store.has_node("test-task-1")

            # Verify dependencies are preserved
            goal_node = new_store.get_node("test-goal-1")
            subgoal_node = new_store.get_node("test-subgoal-1")
            new_store.get_node("test-task-1")

            assert "test-subgoal-1" in goal_node.dependencies
            assert "test-task-1" in subgoal_node.dependencies

        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        store = GraphStore()

        with pytest.raises(FileNotFoundError):
            store.load_graph(Path("nonexistent.json"))

    def test_complex_dependency_scenario(self):
        """Test a complex dependency scenario."""
        store = GraphStore()

        # Create a more complex DAG:
        # Goal -> [SubgoalA, SubgoalB] -> [TaskA, TaskB, TaskC]
        # SubgoalA -> TaskA, TaskB
        # SubgoalB -> TaskC
        # TaskB -> TaskA (TaskB depends on TaskA)

        goal = GoalNode(id="goal", description="Main Goal")
        subgoal_a = GoalNode(id="subgoal_a", description="Subgoal A")
        subgoal_b = GoalNode(id="subgoal_b", description="Subgoal B")
        task_a = GoalNode(id="task_a", description="Task A", status=NodeStatus.COMPLETED)
        task_b = GoalNode(id="task_b", description="Task B", status=NodeStatus.PENDING)
        task_c = GoalNode(id="task_c", description="Task C", status=NodeStatus.PENDING)

        # Add all nodes
        for node in [goal, subgoal_a, subgoal_b, task_a, task_b, task_c]:
            store.add_node(node)

        # Add parent-child relationships
        # Goal is the parent of SubgoalA and SubgoalB
        store.add_parent_child("goal", "subgoal_a")
        store.add_parent_child("goal", "subgoal_b")
        # SubgoalA is the parent of TaskA and TaskB
        store.add_parent_child("subgoal_a", "task_a")
        store.add_parent_child("subgoal_a", "task_b")
        # SubgoalB is the parent of TaskC
        store.add_parent_child("subgoal_b", "task_c")
        # TaskB depends on TaskA (TaskB needs TaskA to be completed first)
        store.add_dependency("task_b", "task_a")

        # Test ready nodes - should be goal, subgoal_a, subgoal_b, task_b, and task_c
        # task_a is completed, so it's not ready
        # All other nodes have no dependencies, so they are ready
        ready_nodes = store.get_ready_nodes()
        ready_ids = {node.id for node in ready_nodes}
        assert ready_ids == {"goal", "subgoal_a", "subgoal_b", "task_b", "task_c"}

        # Test descendants of goal
        # Goal is the parent of SubgoalA and SubgoalB, and they are parents of tasks
        # So goal should have all descendants: subgoal_a, subgoal_b, task_a, task_b, task_c
        descendants = store.get_descendants("goal")
        descendant_ids = set(descendants)
        expected_descendants = {"subgoal_a", "subgoal_b", "task_a", "task_b", "task_c"}
        assert descendant_ids == expected_descendants

        # Test descendants of subgoal_a (should have task_a and task_b)
        descendants = store.get_descendants("subgoal_a")
        descendant_ids = set(descendants)
        expected_descendants = {"task_a", "task_b"}
        assert descendant_ids == expected_descendants

        # Test descendants of subgoal_b (should have task_c)
        descendants = store.get_descendants("subgoal_b")
        descendant_ids = set(descendants)
        expected_descendants = {"task_c"}
        assert descendant_ids == expected_descendants

        # Test ancestors of task_b
        # Task_b depends on task_a, so task_a is an ancestor
        # Task_b is also a child of subgoal_a, so subgoal_a is an ancestor
        # And subgoal_a is a child of goal, so goal is also an ancestor
        ancestors = store.get_ancestors("task_b")
        ancestor_ids = set(ancestors)
        expected_ancestors = {"task_a", "subgoal_a", "goal"}
        assert ancestor_ids == expected_ancestors

        # Test ancestors of goal
        # Goal is the root, so it has no ancestors
        ancestors = store.get_ancestors("goal")
        ancestor_ids = set(ancestors)
        expected_ancestors = set()
        assert ancestor_ids == expected_ancestors

    def test_remove_node_with_dependencies(self):
        """Test removing a node that has dependencies."""
        store = GraphStore()

        node_a = GoalNode(id="a", description="Node A")
        node_b = GoalNode(id="b", description="Node B")
        node_c = GoalNode(id="c", description="Node C")

        store.add_node(node_a)
        store.add_node(node_b)
        store.add_node(node_c)

        # A -> B -> C
        store.add_dependency("a", "b")
        store.add_dependency("b", "c")

        # Remove B (middle node)
        store.remove_node("b")

        # Verify B is gone and dependencies are cleaned up
        assert not store.has_node("b")
        assert "b" not in node_a.dependencies

        # A and C should still exist
        assert store.has_node("a")
        assert store.has_node("c")

    def test_topological_order(self):
        """Test that dependencies create proper topological ordering."""
        store = GraphStore()

        # Create linear chain: A -> B -> C -> D
        nodes = []
        for i, letter in enumerate(["A", "B", "C", "D"]):
            node = GoalNode(id=letter.lower(), description=f"Node {letter}", status=NodeStatus.PENDING)
            nodes.append(node)
            store.add_node(node)

        # Add dependencies: A depends on B, B depends on C, C depends on D
        store.add_dependency("a", "b")
        store.add_dependency("b", "c")
        store.add_dependency("c", "d")

        # Only D should be ready initially (it has no dependencies)
        ready = store.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "d"

        # Complete D, now C should be ready
        store.get_node("d").status = NodeStatus.COMPLETED
        store.update_node(store.get_node("d"))

        ready = store.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "c"
