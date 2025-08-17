"""
Memory integration module for the GoalithService.

Enriches goals with external context and maintains execution history.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.memory import MemoryInterface


class FileSystemStore(MemoryInterface):
    """
    File system-based implementation of MemoryInterface.

    Stores data as JSON files in a directory structure.
    """

    def __init__(self, base_path: Path, name: str = "filesystem"):
        """
        Initialize filesystem store.

        Args:
            base_path: Base directory for storage
            name: Name of this store
        """
        self._name = name
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self._base_path / "data").mkdir(exist_ok=True)
        (self._base_path / "metadata").mkdir(exist_ok=True)

    @property
    def name(self) -> str:
        """Get the name of this store."""
        return self._name

    @property
    def base_path(self) -> Path:
        """Get the base path of this store."""
        return self._base_path

    def _get_data_path(self, key: str) -> Path:
        """Get file path for data."""
        # Replace problematic characters for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._base_path / "data" / f"{safe_key}.json"

    def _get_metadata_path(self, key: str) -> Path:
        """Get file path for metadata."""
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._base_path / "metadata" / f"{safe_key}.json"

    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a value to filesystem."""
        try:
            # Store data
            data_path = self._get_data_path(key)
            with open(data_path, "w") as f:
                json.dump(value, f, indent=2, default=str)

            # Store metadata if provided
            if metadata:
                metadata_path = self._get_metadata_path(key)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Error storing {key}: {e}")
            return False

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from filesystem."""
        try:
            data_path = self._get_data_path(key)
            if not data_path.exists():
                return None

            with open(data_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error retrieving {key}: {e}")
            return None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search values in filesystem (simple filename matching)."""
        results = []
        query_lower = query.lower()

        data_dir = self._base_path / "data"
        if not data_dir.exists():
            return results

        for file_path in data_dir.glob("*.json"):
            if len(results) >= limit:
                break

            key = file_path.stem
            # Check if query matches the key OR the content
            key_match = query_lower in key.lower()

            # Also check if query matches the stored content
            content_match = False
            try:
                value = self.retrieve(key)
                if value is not None:
                    content_match = query_lower in str(value).lower()
            except Exception:
                pass

            if key_match or content_match:
                if value is None:  # Load if not already loaded
                    value = self.retrieve(key)

                if value is not None:
                    result = {"key": key, "value": value}

                    # Try to load metadata
                    metadata_path = self._get_metadata_path(key)
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r") as f:
                                result["metadata"] = json.load(f)
                        except Exception:
                            pass

                    results.append(result)

        return results

    def delete(self, key: str) -> bool:
        """Delete a value from filesystem."""
        try:
            data_path = self._get_data_path(key)
            metadata_path = self._get_metadata_path(key)

            deleted_something = False
            if data_path.exists():
                data_path.unlink()
                deleted_something = True
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_something = True

            return deleted_something
        except Exception as e:
            print(f"Error deleting {key}: {e}")
            return False

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List keys in filesystem."""
        keys = []
        data_dir = self._base_path / "data"

        if not data_dir.exists():
            return keys

        for file_path in data_dir.glob("*.json"):
            key = file_path.stem
            if prefix is None or key.startswith(prefix):
                keys.append(key)

        return keys

    def clear(self) -> bool:
        """Clear all data from filesystem storage."""
        try:
            import shutil

            shutil.rmtree(self._base_path / "data", ignore_errors=True)
            shutil.rmtree(self._base_path / "metadata", ignore_errors=True)
            (self._base_path / "data").mkdir(exist_ok=True)
            (self._base_path / "metadata").mkdir(exist_ok=True)
            return True
        except Exception:
            return False
