"""
Unit tests for goalith.base.memory module.
"""
from typing import Any, Dict, List, Optional

import pytest

from cogents.goalith.base.memory import MemoryInterface


class ConcreteMemoryInterface(MemoryInterface):
    """Concrete implementation for testing abstract base class."""

    def __init__(self):
        self.contexts = {}
        self.execution_histories = {}
        self._data = {}
        self._metadata = {}

    @property
    def name(self) -> str:
        """Get the name of this memory backend."""
        return "concrete_test_memory"

    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a value with optional metadata."""
        try:
            self._data[key] = value
            if metadata:
                self._metadata[key] = metadata
            return True
        except Exception:
            return False

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        return self._data.get(key)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for values matching a query."""
        results = []
        query_lower = query.lower()

        for key, value in self._data.items():
            if len(results) >= limit:
                break

            if query_lower in key.lower() or query_lower in str(value).lower():
                result = {"key": key, "value": value}
                if key in self._metadata:
                    result["metadata"] = self._metadata[key]
                results.append(result)

        return results

    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        try:
            self._data.pop(key, None)
            self._metadata.pop(key, None)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix."""
        if prefix:
            return [key for key in self._data.keys() if key.startswith(prefix)]
        return list(self._data.keys())

    def clear(self) -> bool:
        """Clear all stored data."""
        try:
            self._data.clear()
            self._metadata.clear()
            return True
        except Exception:
            return False

    def store_context(self, node_id, key, context):
        """Store context in memory."""
        if node_id not in self.contexts:
            self.contexts[node_id] = {}
        self.contexts[node_id][key] = context

    def get_context(self, node_id, key):
        """Retrieve context from memory."""
        return self.contexts.get(node_id, {}).get(key)

    def store_execution_note(self, node_id, note):
        """Store execution note."""
        if node_id not in self.execution_histories:
            self.execution_histories[node_id] = []
        self.execution_histories[node_id].append(note)

    def get_execution_history(self, node_id):
        """Get execution history."""
        return self.execution_histories.get(node_id, [])

    def search_nodes(self, query, filters=None, limit=10):
        """Search nodes (simple implementation)."""
        return []


class TestMemoryInterface:
    """Test MemoryInterface abstract base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that MemoryInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryInterface()

    def test_concrete_implementation_store_and_get_context(self):
        """Test storing and retrieving context."""
        memory = ConcreteMemoryInterface()

        context_data = {"key": "value", "number": 42}
        memory.store_context("node1", "test_key", context_data)

        retrieved = memory.get_context("node1", "test_key")
        assert retrieved == context_data

    def test_get_nonexistent_context_returns_none(self):
        """Test getting nonexistent context returns None."""
        memory = ConcreteMemoryInterface()

        result = memory.get_context("nonexistent", "key")
        assert result is None

    def test_store_and_get_execution_history(self):
        """Test storing and retrieving execution history."""
        memory = ConcreteMemoryInterface()

        notes = [
            {"action": "start", "timestamp": "2023-01-01T10:00:00Z"},
            {"action": "progress", "timestamp": "2023-01-01T11:00:00Z"},
        ]

        for note in notes:
            memory.store_execution_note("node1", note)

        history = memory.get_execution_history("node1")
        assert history == notes

    def test_search_nodes_interface(self):
        """Test search nodes interface."""
        memory = ConcreteMemoryInterface()

        # Should not raise error
        results = memory.search_nodes("test query")
        assert isinstance(results, list)

        # With filters
        filters = {"type": "goal", "status": "completed"}
        results = memory.search_nodes("test", filters=filters, limit=5)
        assert isinstance(results, list)

    def test_multiple_contexts_per_node(self):
        """Test storing multiple contexts for same node."""
        memory = ConcreteMemoryInterface()

        memory.store_context("node1", "context1", {"data": "first"})
        memory.store_context("node1", "context2", {"data": "second"})

        assert memory.get_context("node1", "context1") == {"data": "first"}
        assert memory.get_context("node1", "context2") == {"data": "second"}

    def test_contexts_isolated_by_node_id(self):
        """Test that contexts are isolated by node ID."""
        memory = ConcreteMemoryInterface()

        memory.store_context("node1", "shared_key", {"node": "one"})
        memory.store_context("node2", "shared_key", {"node": "two"})

        assert memory.get_context("node1", "shared_key") == {"node": "one"}
        assert memory.get_context("node2", "shared_key") == {"node": "two"}
