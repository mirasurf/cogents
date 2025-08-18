"""
Unit tests for goalith.base.errors module.
"""
import pytest

from cogents.goalith.base.errors import CycleDetectedError, DecompositionError, NodeNotFoundError, SchedulingError


class TestGoalithErrors:
    """Test custom exception classes."""

    def test_decomposition_error_inheritance(self):
        """Test that DecompositionError inherits from Exception."""
        error = DecompositionError("Test message")
        assert isinstance(error, Exception)
        assert str(error) == "Test message"

    def test_cycle_detected_error_inheritance(self):
        """Test that CycleDetectedError inherits from Exception."""
        error = CycleDetectedError("Cycle detected")
        assert isinstance(error, Exception)
        assert str(error) == "Cycle detected"

    def test_node_not_found_error_inheritance(self):
        """Test that NodeNotFoundError inherits from Exception."""
        error = NodeNotFoundError("Node not found")
        assert isinstance(error, Exception)
        assert str(error) == "Node not found"

    def test_scheduling_error_inheritance(self):
        """Test that SchedulingError inherits from Exception."""
        error = SchedulingError("Scheduling failed")
        assert isinstance(error, Exception)
        assert str(error) == "Scheduling failed"

    def test_errors_can_be_raised_and_caught(self):
        """Test that custom errors can be raised and caught properly."""
        # Test DecompositionError
        with pytest.raises(DecompositionError) as exc_info:
            raise DecompositionError("Decomposition failed")
        assert str(exc_info.value) == "Decomposition failed"

        # Test CycleDetectedError
        with pytest.raises(CycleDetectedError) as exc_info:
            raise CycleDetectedError("Cycle in graph")
        assert str(exc_info.value) == "Cycle in graph"

        # Test NodeNotFoundError
        with pytest.raises(NodeNotFoundError) as exc_info:
            raise NodeNotFoundError("Missing node")
        assert str(exc_info.value) == "Missing node"

        # Test SchedulingError
        with pytest.raises(SchedulingError) as exc_info:
            raise SchedulingError("Schedule conflict")
        assert str(exc_info.value) == "Schedule conflict"
