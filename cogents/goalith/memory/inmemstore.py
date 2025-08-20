from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base.memory import MemoryInterface


class InMemoryStore(MemoryInterface):
    """
    Simple in-memory implementation of MemoryInterface.

    Suitable for development and testing.
    """

    def __init__(self, name: str = "in_memory"):
        """
        Initialize in-memory store.

        Args:
            name: Name of this store
        """
        self._name = name
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        """Get the name of this store."""
        return self._name

    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a value in memory."""
        try:
            self._data[key] = value
            if metadata:
                self._metadata[key] = metadata
            return True
        except Exception:
            return False

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        return self._data.get(key)

    def get(self, key: str) -> Optional[Any]:
        """Alias for retrieve method."""
        return self.retrieve(key)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search values in memory (simple string matching)."""
        results = []
        query_lower = query.lower()

        for key, value in self._data.items():
            if len(results) >= limit:
                break

            # Simple search: check if query appears in key or string representation of value
            if query_lower in key.lower() or query_lower in str(value).lower():
                result = {"key": key, "value": value}
                if key in self._metadata:
                    result["metadata"] = self._metadata[key]
                results.append(result)

        return results

    def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        try:
            self._data.pop(key, None)
            self._metadata.pop(key, None)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List keys in memory."""
        if prefix:
            return [key for key in self._data.keys() if key.startswith(prefix)]
        return list(self._data.keys())

    def clear(self) -> bool:
        """Clear all data."""
        try:
            self._data.clear()
            self._metadata.clear()
            return True
        except Exception:
            return False

    def size(self) -> int:
        """Get number of stored items."""
        return len(self._data)

    def store_context(self, node_id: str, context_key: str, context_value: Any) -> bool:
        """Store context for a node."""
        key = f"context:{node_id}:{context_key}"
        return self.store(key, context_value, {"node_id": node_id})

    def store_execution_note(self, node_id: str, note: Dict[str, Any]) -> bool:
        """Store an execution note for a node."""
        # Get existing history or create new list
        history_key = f"history:{node_id}"
        existing_history = self.retrieve(history_key) or []

        # Ensure it's a list
        if not isinstance(existing_history, list):
            existing_history = [existing_history] if existing_history else []

        # Add new note
        existing_history.append(note)

        # Store updated history
        metadata = {
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self.store(history_key, existing_history, metadata)

    def get_context(self, node_id: str, context_key: str) -> Optional[Any]:
        """Get context for a node."""
        key = f"context:{node_id}:{context_key}"
        return self.get(key)

    def get_execution_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a node."""
        # Get the history list for this node
        history_key = f"history:{node_id}"
        history = self.get(history_key)

        if history is None:
            return []

        # Ensure it's always a list
        if not isinstance(history, list):
            return [history]

        return history

    def is_full(self) -> bool:
        """Check if the store is full."""
        return False  # In-memory store is never full
