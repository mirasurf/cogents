from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryInterface(ABC):
    """
    Abstract interface for memory backends.

    Can be implemented by vector databases, SQL databases, in-memory caches, etc.
    """

    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with optional metadata.

        Args:
            key: Storage key
            value: Value to store
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.

        Args:
            key: Storage key

        Returns:
            Retrieved value or None if not found
        """

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for values matching a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results with keys and values
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value by key.

        Args:
            key: Storage key

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all keys, optionally filtered by prefix.

        Args:
            prefix: Optional key prefix filter

        Returns:
            List of matching keys
        """

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all stored data.

        Returns:
            True if successful, False otherwise
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this memory backend."""
