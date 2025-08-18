"""
Unit tests for goalith.base.update_event module.
"""
import pytest
from datetime import datetime, timezone
from uuid import UUID

from cogents.goalith.base.update_event import UpdateEvent, UpdateType


class TestUpdateType:
    """Test UpdateType enum."""

    def test_update_type_values(self):
        """Test that UpdateType has correct values."""
        assert UpdateType.STATUS_CHANGE == "status_change"
        assert UpdateType.NODE_EDIT == "node_edit"
        assert UpdateType.NODE_ADD == "node_add"
        assert UpdateType.NODE_REMOVE == "node_remove"
        assert UpdateType.DEPENDENCY_ADD == "dependency_add"
        assert UpdateType.DEPENDENCY_REMOVE == "dependency_remove"
        assert UpdateType.CONTEXT_UPDATE == "context_update"
        assert UpdateType.PRIORITY_CHANGE == "priority_change"

    def test_update_type_string_representation(self):
        """Test string representation of UpdateType."""
        assert str(UpdateType.STATUS_CHANGE) == "status_change"
        assert str(UpdateType.NODE_EDIT) == "node_edit"
        assert str(UpdateType.NODE_ADD) == "node_add"


class TestUpdateEvent:
    """Test UpdateEvent data model."""

    def test_minimal_creation(self):
        """Test creating UpdateEvent with minimal parameters."""
        event = UpdateEvent(
            update_type=UpdateType.STATUS_CHANGE,
            node_id="test-node-1"
        )
        
        # Check generated ID
        assert event.id is not None
        assert isinstance(UUID(event.id), UUID)  # Valid UUID format
        
        # Check required fields
        assert event.update_type == UpdateType.STATUS_CHANGE
        assert event.node_id == "test-node-1"
        
        # Check defaults
        assert event.data == {}
        assert event.source is None
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_full_creation(self):
        """Test creating UpdateEvent with all parameters."""
        custom_id = "custom-event-123"
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {"old_status": "pending", "new_status": "in_progress"}
        
        event = UpdateEvent(
            id=custom_id,
            update_type=UpdateType.STATUS_CHANGE,
            node_id="test-node-1",
            data=data,
            source="test-agent",
            timestamp=timestamp
        )
        
        assert event.id == custom_id
        assert event.update_type == UpdateType.STATUS_CHANGE
        assert event.node_id == "test-node-1"
        assert event.data == data
        assert event.source == "test-agent"
        assert event.timestamp == timestamp

    def test_equality_excludes_timestamp(self):
        """Test that equality comparison excludes timestamp."""
        base_event = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "value"},
            source="agent"
        )
        
        # Same event with different timestamp
        different_timestamp_event = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "value"},
            source="agent",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        
        assert base_event == different_timestamp_event

    def test_equality_considers_all_other_fields(self):
        """Test that equality considers all fields except timestamp."""
        base_event = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "value"},
            source="agent"
        )
        
        # Different ID
        different_id = UpdateEvent(
            id="test-2",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "value"},
            source="agent"
        )
        assert base_event != different_id
        
        # Different update_type
        different_type = UpdateEvent(
            id="test-1",
            update_type=UpdateType.NODE_EDIT,
            node_id="node-1",
            data={"key": "value"},
            source="agent"
        )
        assert base_event != different_type
        
        # Different node_id
        different_node = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-2",
            data={"key": "value"},
            source="agent"
        )
        assert base_event != different_node
        
        # Different data
        different_data = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "different"},
            source="agent"
        )
        assert base_event != different_data
        
        # Different source
        different_source = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"key": "value"},
            source="different-agent"
        )
        assert base_event != different_source

    def test_equality_with_non_update_event(self):
        """Test equality with non-UpdateEvent objects."""
        event = UpdateEvent(
            update_type=UpdateType.STATUS_CHANGE,
            node_id="test-node"
        )
        
        assert event != "not-an-event"
        assert event != {"id": event.id}
        assert event != None

    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = UpdateEvent(
            id="test-1",
            update_type=UpdateType.STATUS_CHANGE,
            node_id="node-1",
            data={"old": "pending", "new": "in_progress"},
            source="test-agent",
            timestamp=timestamp
        )
        
        result = event.to_dict()
        
        expected = {
            "id": "test-1",
            "update_type": "status_change",
            "node_id": "node-1",
            "data": {"old": "pending", "new": "in_progress"},
            "source": "test-agent",
            "timestamp": timestamp.isoformat()
        }
        
        assert result == expected

    def test_to_dict_with_defaults(self):
        """Test to_dict with default values."""
        event = UpdateEvent(
            update_type=UpdateType.NODE_ADD,
            node_id="node-1"
        )
        
        result = event.to_dict()
        
        assert result["update_type"] == "node_add"
        assert result["node_id"] == "node-1"
        assert result["data"] == {}
        assert result["source"] is None
        assert "timestamp" in result
        assert "id" in result

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "id": "test-1",
            "update_type": "status_change",
            "node_id": "node-1",
            "data": {"old": "pending", "new": "in_progress"},
            "source": "test-agent",
            "timestamp": "2023-01-01T12:00:00+00:00"
        }
        
        event = UpdateEvent.from_dict(data)
        
        assert event.id == "test-1"
        assert event.update_type == UpdateType.STATUS_CHANGE
        assert event.node_id == "node-1"
        assert event.data == {"old": "pending", "new": "in_progress"}
        assert event.source == "test-agent"
        assert event.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "id": "test-1",
            "update_type": "node_add",
            "node_id": "node-1"
        }
        
        event = UpdateEvent.from_dict(data)
        
        assert event.id == "test-1"
        assert event.update_type == UpdateType.NODE_ADD
        assert event.node_id == "node-1"
        assert event.data == {}
        assert event.source is None
        assert event.timestamp is not None

    def test_from_dict_without_timestamp(self):
        """Test from_dict when timestamp is not provided."""
        data = {
            "id": "test-1",
            "update_type": "node_edit",
            "node_id": "node-1",
            "data": {"field": "value"}
        }
        
        event = UpdateEvent.from_dict(data)
        
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works correctly."""
        original = UpdateEvent(
            id="test-roundtrip",
            update_type=UpdateType.PRIORITY_CHANGE,
            node_id="node-1",
            data={"old_priority": 1.0, "new_priority": 5.0},
            source="scheduler",
            timestamp=datetime(2023, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        )
        
        # Convert to dict and back
        dict_repr = original.to_dict()
        reconstructed = UpdateEvent.from_dict(dict_repr)
        
        # Should be equal (excluding timestamp precision issues)
        assert reconstructed.id == original.id
        assert reconstructed.update_type == original.update_type
        assert reconstructed.node_id == original.node_id
        assert reconstructed.data == original.data
        assert reconstructed.source == original.source
        assert reconstructed.timestamp == original.timestamp

    def test_enum_serialization(self):
        """Test that UpdateType enum is properly serialized."""
        event = UpdateEvent(
            update_type=UpdateType.DEPENDENCY_ADD,
            node_id="node-1"
        )
        
        # Should serialize enum as string
        dict_repr = event.to_dict()
        assert dict_repr["update_type"] == "dependency_add"
        
        # Should deserialize string back to enum
        reconstructed = UpdateEvent.from_dict(dict_repr)
        assert reconstructed.update_type == UpdateType.DEPENDENCY_ADD
        assert isinstance(reconstructed.update_type, UpdateType)