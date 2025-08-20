"""
Unit tests for goalith.decomposer.registry module.
"""
from unittest.mock import Mock

import pytest

from cogents.goalith.base.decomposer import GoalDecomposer
from cogents.goalith.base.goal_node import GoalNode
from cogents.goalith.decomposer.registry import DecomposerRegistry
from cogents.goalith.decomposer.simple_decomposer import SimpleListDecomposer


class TestDecomposerRegistry:
    """Test DecomposerRegistry functionality."""

    def test_empty_initialization(self):
        """Test creating empty registry."""
        registry = DecomposerRegistry()

        assert len(registry._decomposers) == 0
        assert registry.list_decomposers() == []

    def test_register_decomposer(self):
        """Test registering a decomposer."""
        registry = DecomposerRegistry()
        decomposer = SimpleListDecomposer(["task1", "task2"])

        registry.register("simple", decomposer)

        assert registry.has_decomposer("simple")
        assert registry.get_decomposer("simple") is decomposer

    def test_register_duplicate_name_replaces(self):
        """Test that registering with duplicate name replaces existing."""
        registry = DecomposerRegistry()

        decomposer1 = SimpleListDecomposer(["task1", "task2"])
        decomposer2 = SimpleListDecomposer(["task3", "task4"])

        registry.register("duplicate", decomposer1)
        registry.register("duplicate", decomposer2, replace=True)

        assert registry.get_decomposer("duplicate") is decomposer2

    def test_get_decomposer(self):
        """Test getting a registered decomposer."""
        registry = DecomposerRegistry()
        decomposer = SimpleListDecomposer(["task1", "task2"])

        registry.register("test", decomposer)

        retrieved = registry.get_decomposer("test")
        assert retrieved is decomposer

    def test_get_nonexistent_decomposer_raises_error(self):
        """Test that getting nonexistent decomposer raises ValueError."""
        registry = DecomposerRegistry()

        with pytest.raises(ValueError, match="Decomposer 'nonexistent' not found"):
            registry.get_decomposer("nonexistent")

    def test_has_decomposer(self):
        """Test checking if decomposer exists."""
        registry = DecomposerRegistry()
        decomposer = SimpleListDecomposer(["task1", "task2"])

        assert registry.has_decomposer("test") is False

        registry.register("test", decomposer)
        assert registry.has_decomposer("test") is True

    def test_unregister_decomposer(self):
        """Test unregistering a decomposer."""
        registry = DecomposerRegistry()
        decomposer = SimpleListDecomposer(["task1", "task2"])

        registry.register("test", decomposer)
        assert registry.has_decomposer("test")

        registry.unregister("test")
        assert not registry.has_decomposer("test")
        assert "test" not in registry.list_decomposers()

    def test_unregister_nonexistent_decomposer_raises_error(self):
        """Test that unregistering nonexistent decomposer raises ValueError."""
        registry = DecomposerRegistry()

        with pytest.raises(ValueError, match="Decomposer 'nonexistent' not found"):
            registry.unregister("nonexistent")

    def test_list_decomposers(self):
        """Test listing all registered decomposers."""
        registry = DecomposerRegistry()

        # Empty registry
        assert registry.list_decomposers() == []

        # Add some decomposers
        decomposer1 = SimpleListDecomposer(["task1", "task2"])
        decomposer2 = SimpleListDecomposer(["task3", "task4"])

        registry.register("first", decomposer1)
        registry.register("second", decomposer2)

        decomposer_names = registry.list_decomposers()
        assert set(decomposer_names) == {"first", "second"}

    def test_decompose_goal_with_registered_decomposer(self):
        """Test decomposing a goal with a registered decomposer."""
        registry = DecomposerRegistry()

        # Mock decomposer
        mock_decomposer = Mock(spec=GoalDecomposer)
        mock_decomposer.decompose.return_value = []

        registry.register("mock", mock_decomposer)

        goal = GoalNode(description="Test goal")
        context = {"domain": "software"}

        result = registry.decompose("mock", goal, context=context)

        # Verify context was passed correctly
        mock_decomposer.decompose.assert_called_once()
        call_args = mock_decomposer.decompose.call_args
        assert call_args[0][0] == goal  # First positional arg is goal
        assert call_args[0][1] == context  # Second positional arg is context
        assert result == []

    def test_decompose_goal_with_nonexistent_decomposer_raises_error(self):
        """Test that decomposing with nonexistent decomposer raises error."""
        registry = DecomposerRegistry()
        goal = GoalNode(description="Test goal")

        with pytest.raises(ValueError, match="Decomposer 'nonexistent' not found"):
            registry.decompose("nonexistent", goal)

    def test_decompose_goal_propagates_decomposer_errors(self):
        """Test that decomposer errors are properly propagated."""
        registry = DecomposerRegistry()

        # Mock decomposer that raises an error
        failing_decomposer = Mock(spec=GoalDecomposer)
        failing_decomposer.decompose.side_effect = Exception("Decomposition failed")

        registry.register("failing", failing_decomposer)

        goal = GoalNode(description="Test goal")

        with pytest.raises(Exception, match="Decomposition failed"):
            registry.decompose("failing", goal)

    def test_get_decomposer_info(self):
        """Test getting information about a decomposer."""
        registry = DecomposerRegistry()

        # Create a decomposer with some metadata
        decomposer = SimpleListDecomposer(["task1", "task2"])
        registry.register("test", decomposer)

        info = registry.get_decomposer_info("test")

        assert isinstance(info, dict)
        assert info["name"] == "test"
        assert info["type"] == type(decomposer).__name__
        assert "description" in info or "capabilities" in info

    def test_get_info_for_nonexistent_decomposer_raises_error(self):
        """Test that getting info for nonexistent decomposer raises error."""
        registry = DecomposerRegistry()

        with pytest.raises(ValueError):
            registry.get_decomposer_info("nonexistent")

    def test_multiple_decomposer_types(self):
        """Test registry with multiple types of decomposers."""
        registry = DecomposerRegistry()

        # Create different types of decomposers
        simple_decomposer = SimpleListDecomposer(["task1", "task2"])

        # Mock other types
        mock_llm_decomposer = Mock(spec=GoalDecomposer)
        mock_human_decomposer = Mock(spec=GoalDecomposer)

        # Register them
        registry.register("simple", simple_decomposer)
        registry.register("llm", mock_llm_decomposer)
        registry.register("human", mock_human_decomposer)

        # Verify all are registered
        decomposer_names = registry.list_decomposers()
        assert set(decomposer_names) == {"simple", "llm", "human"}

        # Verify each can be retrieved
        assert registry.get_decomposer("simple") is simple_decomposer
        assert registry.get_decomposer("llm") is mock_llm_decomposer
        assert registry.get_decomposer("human") is mock_human_decomposer

    def test_clear_registry(self):
        """Test clearing all decomposers from registry."""
        registry = DecomposerRegistry()

        # Add some decomposers
        registry.register("first", SimpleListDecomposer(["task1", "task2"]))
        registry.register("second", SimpleListDecomposer(["task3", "task4"]))

        assert len(registry.list_decomposers()) == 2

        # Clear registry
        registry.clear()

        assert len(registry.list_decomposers()) == 0
        assert not registry.has_decomposer("first")
        assert not registry.has_decomposer("second")

    def test_registry_as_context_manager(self):
        """Test using registry as context manager for temporary registration."""
        registry = DecomposerRegistry()

        # Register permanent decomposer
        permanent = SimpleListDecomposer(["task1", "task2"])
        registry.register("permanent", permanent)

        # Use context manager for temporary decomposer
        temporary = SimpleListDecomposer(["task3", "task4"])

        with registry.temporary_decomposer("temp", temporary):
            # Both should be available
            assert registry.has_decomposer("permanent")
            assert registry.has_decomposer("temp")
            assert registry.get_decomposer("temp") is temporary

        # Temporary should be gone after context
        assert registry.has_decomposer("permanent")
        assert not registry.has_decomposer("temp")

    def test_decompose_with_context_parameter_passing(self):
        """Test that context parameters are properly passed to decomposer."""
        registry = DecomposerRegistry()

        # Mock decomposer that inspects context
        mock_decomposer = Mock(spec=GoalDecomposer)
        mock_decomposer.decompose.return_value = []

        registry.register("context_test", mock_decomposer)

        goal = GoalNode(description="Test goal")
        context = {
            "domain": "software",
            "user_preferences": {"style": "agile"},
            "historical_data": ["pattern1", "pattern2"],
        }

        # Decompose with context
        result = registry.decompose("context_test", goal, context=context)

        # Verify context was passed correctly
        mock_decomposer.decompose.assert_called_once()
        call_args = mock_decomposer.decompose.call_args
        assert call_args[0][0] == goal  # First positional arg is goal
        assert call_args[0][1] == context  # Second positional arg is context
        assert result == []

    def test_bulk_registration(self):
        """Test registering multiple decomposers at once."""
        registry = DecomposerRegistry()

        decomposers = {
            "simple1": SimpleListDecomposer(["task1", "task2"]),
            "simple2": SimpleListDecomposer(["task3", "task4"]),
            "simple3": SimpleListDecomposer(["task5", "task6"]),
        }

        # Bulk register
        registry.register_bulk(decomposers)

        # Verify all are registered
        for name in decomposers:
            assert registry.has_decomposer(name)
            assert registry.get_decomposer(name) is decomposers[name]

    def test_registry_statistics(self):
        """Test getting registry statistics and usage info."""
        registry = DecomposerRegistry()

        # Add decomposers
        registry.register("decomposer1", SimpleListDecomposer(["task1", "task2"]))
        registry.register("decomposer2", SimpleListDecomposer(["task3", "task4"]))

        # Use them
        goal = GoalNode(description="Test")
        mock_decomposer = Mock(spec=GoalDecomposer)
        mock_decomposer.decompose.return_value = []
        registry.register("mock", mock_decomposer)

        registry.decompose("mock", goal)
        registry.decompose("mock", goal)

        # Get statistics
        stats = registry.get_registry_info()
        assert len(stats) == 3

    def test_decomposer_validation(self):
        """Test validation of decomposer instances."""
        registry = DecomposerRegistry()

        # Valid decomposer
        valid_decomposer = SimpleListDecomposer(["task1", "task2"])
        registry.register("valid", valid_decomposer)  # Should not raise

        # Invalid decomposer (not implementing interface)
        with pytest.raises(TypeError):
            registry.register("invalid", "not_a_decomposer")

    def test_concurrent_access(self):
        """Test concurrent access to registry."""
        import threading

        registry = DecomposerRegistry()
        errors = []

        def register_decomposer(name_prefix, count):
            try:
                for i in range(count):
                    decomposer = SimpleListDecomposer([f"task{i}"])
                    registry.register(f"{name_prefix}_{i}", decomposer)
            except Exception as e:
                errors.append(e)

        def use_decomposer(name):
            try:
                if registry.has_decomposer(name):
                    goal = GoalNode(description="Concurrent test")
                    registry.decompose(name, goal)
            except Exception as e:
                errors.append(e)

        # Register some initial decomposers
        for i in range(5):
            registry.register(f"initial_{i}", SimpleListDecomposer([f"task{i}"]))

        # Create threads for concurrent access
        threads = []

        # Registration threads
        for i in range(3):
            thread = threading.Thread(target=register_decomposer, args=(f"thread_{i}", 5))
            threads.append(thread)

        # Usage threads
        for i in range(5):
            thread = threading.Thread(target=use_decomposer, args=(f"initial_{i}",))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify expected decomposers are registered
        decomposer_names = registry.list_decomposers()
        assert len(decomposer_names) >= 20  # 5 initial + 3*5 from threads
