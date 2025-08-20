from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base.goal_node import GoalNode
from ..base.memory import MemoryInterface
from .inmemstore import InMemoryStore


class MemoryManager:
    """
    Manages memory operations for the GoalithService.

    Retrieves stored context on node creation/lookup and persists
    annotations, execution notes, and performance metrics.
    """

    def __init__(self, memory_backend: Optional[MemoryInterface] = None):
        """
        Initialize memory manager.

        Args:
            memory_backend: Memory backend to use (default: in-memory)
        """
        self._backend = memory_backend or InMemoryStore()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_timeout = timedelta(minutes=30)

        # Statistics
        self._stats = {
            "context_retrievals": 0,
            "context_stores": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "enrichments": 0,
            "searches": 0,
        }

    @property
    def backend(self) -> MemoryInterface:
        """Get the memory backend."""
        return self._backend

    def enrich_node_context(self, node: GoalNode) -> Dict[str, Any]:
        """
        Enrich a node with relevant context from memory.

        Args:
            node: The node to enrich

        Returns:
            Dictionary of enriched context
        """
        self._stats["enrichments"] += 1
        enriched_context = {
            "node_id": node.id,  # Include node ID as tests expect
            "description": node.description,
            "type": node.type,
        }

        # Look for stored context for this specific node
        node_context_keys = self._backend.list_keys(f"context:{node.id}:")
        for key in node_context_keys:
            parts = key.split(":", 2)
            if len(parts) >= 3:
                context_key = parts[2]  # Extract the context key part
                context_value = self._backend.get(key)
                if context_value is not None:
                    # If the context value is a dict, flatten it into the enriched context
                    if isinstance(context_value, dict):
                        enriched_context.update(context_value)
                    else:
                        enriched_context[context_key] = context_value

        # Search for related context based on node description and tags
        search_terms = [node.description]
        search_terms.extend(node.tags)

        for term in search_terms:
            if len(term) < 3:  # Skip very short terms
                continue

            results = self._search_memory(term, limit=5)
            for result in results:
                key = result["key"]
                if key.startswith("context:"):
                    enriched_context[key] = result["value"]

        # Look for specific patterns
        enriched_context.update(self._get_pattern_context(node))

        return enriched_context

    def store_node_context(self, node: GoalNode, context_key: str, context_value: Any) -> bool:
        """
        Store context for a node.

        Args:
            node: The node
            context_key: Key for the context
            context_value: Context value

        Returns:
            True if successful, False otherwise
        """
        self._stats["context_stores"] += 1

        full_key = f"context:{node.id}:{context_key}"
        metadata = {
            "node_id": node.id,
            "node_type": node.type,
            "node_description": node.description,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": list(node.tags),
        }

        return self._backend.store(full_key, context_value, metadata)

    def store_context(self, node_id: str, context_key: str, context_value: Any) -> bool:
        """
        Store context for a node by ID.

        Args:
            node_id: The node ID
            context_key: Key for the context
            context_value: Context value

        Returns:
            True if successful, False otherwise
        """
        self._stats["context_stores"] += 1

        full_key = f"context:{node_id}:{context_key}"
        metadata = {
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._backend.store(full_key, context_value, metadata)

    def get_context(self, node_id: str, context_key: str) -> Optional[Any]:
        """
        Get context for a node by ID.

        Args:
            node_id: The node ID
            context_key: Key for the context

        Returns:
            Context value if found, None otherwise
        """
        self._stats["context_retrievals"] += 1

        full_key = f"context:{node_id}:{context_key}"

        # Check cache first
        if self._is_cached(full_key):
            self._stats["cache_hits"] += 1
            return self._cache[full_key]

        # Retrieve from backend
        self._stats["cache_misses"] += 1
        result = self._backend.get(full_key)

        if result is not None:
            self._cache_value(full_key, result)

        return result

    def store_execution_note(self, node_id: str, note: Dict[str, Any]) -> bool:
        """
        Store an execution note for a node.

        Args:
            node_id: The node ID
            note: The execution note to store

        Returns:
            True if successful, False otherwise
        """
        self._stats["context_stores"] += 1

        # Use the backend's store_execution_note method if available
        if hasattr(self._backend, "store_execution_note"):
            return self._backend.store_execution_note(node_id, note)

        # Fallback to manual implementation
        history_key = f"history:{node_id}"
        existing_history = self._backend.retrieve(history_key) or []

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

        return self._backend.store(history_key, existing_history, metadata)

    def get_node_context(self, node: GoalNode, context_key: str) -> Optional[Any]:
        """
        Get stored context for a node.

        Args:
            node: The node
            context_key: Key for the context

        Returns:
            Context value or None if not found
        """
        self._stats["context_retrievals"] += 1

        full_key = f"context:{node.id}:{context_key}"

        # Check cache first
        if self._is_cached(full_key):
            self._stats["cache_hits"] += 1
            return self._cache[full_key]

        # Retrieve from backend
        self._stats["cache_misses"] += 1
        value = self._backend.retrieve(full_key)

        if value is not None:
            self._cache_value(full_key, value)

        return value

    def store_execution_history(self, node_or_id, execution_data=None) -> bool:
        """
        Store execution history for a node.

        Args:
            node_or_id: The node or node ID to store history for
            execution_data: Optional execution data dict (when node_or_id is string)

        Returns:
            True if successful, False otherwise
        """
        if isinstance(node_or_id, str):
            # Called with (node_id, execution_data) format
            node_id = node_or_id
            new_entry = {
                "result": execution_data or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            metadata = {
                "type": "execution_history",
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            # Called with GoalNode format
            node = node_or_id
            node_id = node.id
            new_entry = {
                "result": {
                    "node_id": node.id,
                    "description": node.description,
                    "type": node.type,
                    "status": node.status,
                    "created_at": node.created_at.isoformat(),
                    "updated_at": node.updated_at.isoformat(),
                    "started_at": node.started_at.isoformat() if node.started_at else None,
                    "completed_at": node.completed_at.isoformat() if node.completed_at else None,
                    "retry_count": node.retry_count,
                    "execution_notes": node.execution_notes,
                    "performance_metrics": node.performance_metrics,
                    "final_context": node.context,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
            metadata = {
                "type": "execution_history",
                "node_type": node.type,
                "final_status": node.status,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Get existing history or create new list
        history_key = f"history:{node_id}"
        existing_history = self._backend.retrieve(history_key) or []

        # Ensure it's a list
        if not isinstance(existing_history, list):
            existing_history = [existing_history] if existing_history else []

        # Add new entry
        existing_history.append(new_entry)

        return self._backend.store(history_key, existing_history, metadata)

    def get_execution_history(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get execution history for a node.

        Args:
            node_id: ID of the node

        Returns:
            List of execution history entries
        """
        history_key = f"history:{node_id}"
        history = self._backend.retrieve(history_key)

        if history is None:
            return []

        # Ensure it's always a list
        if not isinstance(history, list):
            return [history]

        return history

    def store_performance_metrics(self, node_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Store performance metrics for a node.

        Args:
            node_id: ID of the node
            metrics: Performance metrics

        Returns:
            True if successful, False otherwise
        """
        metrics_key = f"metrics:{node_id}"
        timestamped_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }

        metadata = {
            "type": "performance_metrics",
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._backend.store(metrics_key, timestamped_metrics, metadata)

    def get_similar_nodes(self, node: GoalNode, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find nodes similar to the given node.

        Args:
            node: The reference node
            limit: Maximum number of results

        Returns:
            List of similar nodes with their history
        """
        # Search for nodes with similar descriptions or tags
        search_results = []

        # Search by description
        desc_results = self._search_memory(node.description, limit=limit * 2)
        search_results.extend(desc_results)

        # Search by tags
        for tag in node.tags:
            tag_results = self._search_memory(tag, limit=limit)
            search_results.extend(tag_results)

        # Filter for history entries and remove duplicates
        similar_nodes = []
        seen_ids = set()

        for result in search_results:
            if len(similar_nodes) >= limit:
                break

            if result["key"].startswith("history:"):
                node_id = result["key"].split(":", 1)[1]
                if node_id not in seen_ids and node_id != node.id:
                    seen_ids.add(node_id)
                    similar_nodes.append(result)

        return similar_nodes

    def search_similar_goals(
        self, node: GoalNode, filters: Optional[Dict[str, Any]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar goals based on description and tags.

        Args:
            node: The node to find similar goals for
            filters: Optional filters to apply
            limit: Maximum number of results to return

        Returns:
            List of similar goal contexts
        """
        self._stats["searches"] += 1

        search_terms = [node.description]
        search_terms.extend(node.tags)

        results = []
        for term in search_terms:
            if len(term) < 3:
                continue

            term_results = self._search_memory(term, limit=limit)
            results.extend(term_results)

        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                if self._matches_filters(result, filters):
                    filtered_results.append(result)
            results = filtered_results

        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            if result["key"] not in seen:
                seen.add(result["key"])
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break

        return unique_results

    def cleanup_old_data(self, max_age: timedelta = timedelta(days=30)) -> int:
        """
        Clean up old data from memory.

        Args:
            max_age: Maximum age for data retention

        Returns:
            Number of items cleaned up
        """
        cutoff_time = datetime.utcnow() - max_age
        cleaned_count = 0

        # Get all keys
        all_keys = self._backend.list_keys()

        for key in all_keys:
            # Check if this is timestamped data
            value = self._backend.retrieve(key)
            if isinstance(value, dict) and "timestamp" in value:
                try:
                    timestamp = datetime.fromisoformat(value["timestamp"])
                    if timestamp < cutoff_time:
                        if self._backend.delete(key):
                            cleaned_count += 1
                except (ValueError, KeyError):
                    pass  # Skip invalid timestamps

        # Clean cache as well
        self._cleanup_cache()

        return cleaned_count

    def _search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory with caching."""
        cache_key = f"search:{query}:{limit}"

        if self._is_cached(cache_key):
            return self._cache[cache_key]

        results = self._backend.search(query, limit)
        self._cache_value(cache_key, results)

        return results

    def _get_pattern_context(self, node: GoalNode) -> Dict[str, Any]:
        """Get context based on common patterns."""
        context = {}

        # Add type-specific context
        if node.type in ["goal", "subgoal"]:
            context["planning_context"] = {
                "decomposition_strategies": ["sequential", "parallel", "hybrid"],
                "common_pitfalls": ["over_decomposition", "unclear_dependencies"],
            }
        elif node.type == "task":
            context["execution_context"] = {
                "retry_strategies": ["exponential_backoff", "circuit_breaker"],
                "monitoring_points": ["start", "progress", "completion"],
            }

        # Add deadline-related context
        if node.deadline:
            time_to_deadline = node.deadline - datetime.utcnow()
            context["deadline_context"] = {
                "urgency_level": "high" if time_to_deadline.days < 1 else "normal",
                "time_remaining": time_to_deadline.total_seconds(),
            }

        return context

    def _is_cached(self, key: str) -> bool:
        """Check if a value is cached and still valid."""
        if key not in self._cache:
            return False

        if key in self._cache_ttl:
            if datetime.utcnow() > self._cache_ttl[key]:
                # Cache expired
                del self._cache[key]
                del self._cache_ttl[key]
                return False

        return True

    def _cache_value(self, key: str, value: Any) -> None:
        """Cache a value with TTL."""
        self._cache[key] = value
        self._cache_ttl[key] = datetime.utcnow() + self._cache_timeout

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        now = datetime.utcnow()
        expired_keys = [key for key, expiry in self._cache_ttl.items() if now > expiry]

        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_ttl.pop(key, None)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_ttl.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Memory statistics
        """
        return {
            "backend": self._backend.name,
            "cache_size": len(self._cache),
            "operations": self._stats.copy(),
            "cache_hit_rate": (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0
            ),
        }

    # Alias methods expected by tests
    def store_goal_context(self, goal_id: str, context: Dict[str, Any]) -> bool:
        """
        Store context for a goal by ID.

        Args:
            goal_id: The goal ID
            context: Context dict to store

        Returns:
            True if successful, False otherwise
        """
        self._stats["context_stores"] += 1

        full_key = f"context:{goal_id}"
        metadata = {
            "goal_id": goal_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._backend.store(full_key, context, metadata)

    def get_goal_context(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored context for a goal by ID.

        Args:
            goal_id: The goal ID

        Returns:
            Stored context or None if not found
        """
        self._stats["context_retrievals"] += 1

        # Check cache first
        if goal_id in self._cache:
            if goal_id in self._cache_ttl and datetime.utcnow() < self._cache_ttl[goal_id]:
                self._stats["cache_hits"] += 1
                return self._cache[goal_id]

        self._stats["cache_misses"] += 1

        full_key = f"context:{goal_id}"
        result = self._backend.retrieve(full_key)

        if result:
            # Cache the result
            self._cache[goal_id] = result
            self._cache_ttl[goal_id] = datetime.utcnow() + self._cache_timeout

        return result

    def enrich_goal(self, goal: GoalNode) -> GoalNode:
        """
        Enrich a goal with context and return an updated copy.

        Args:
            goal: The goal to enrich

        Returns:
            Goal with enriched context
        """
        # Get stored context for this goal
        stored_context = self.get_goal_context(goal.id)

        # Get related context based on description/tags
        enriched_context = self.enrich_node_context(goal)

        # Combine contexts
        final_context = goal.context.copy() if goal.context else {}

        if stored_context:
            final_context.update(stored_context)

        if enriched_context:
            final_context["memory_context"] = enriched_context

        # Create a new goal with enriched context - use estimated_effort instead of estimated_duration
        enriched_goal = GoalNode(
            id=goal.id,
            description=goal.description,
            priority=goal.priority,
            status=goal.status,
            type=goal.type,
            tags=goal.tags,
            context=final_context,
            dependencies=goal.dependencies,
            deadline=goal.deadline,
            estimated_effort=goal.estimated_effort,  # Fixed: was estimated_duration
            retry_count=goal.retry_count,
            parent=goal.parent,  # Fixed: was parent_id
            children=goal.children,
            created_at=goal.created_at,
            updated_at=goal.updated_at,
            completed_at=goal.completed_at,
        )

        return enriched_goal

    # Keep original methods for backward compatibility
    def store_node_context_alt(self, node: GoalNode, context_key: str, context_value: Any) -> bool:
        """Original store_node_context method."""
        return self.store_node_context(node, context_key, context_value)

    def get_node_context_alt(self, node: GoalNode, context_key: str) -> Optional[Any]:
        """Original get_node_context method."""
        return self.get_node_context(node, context_key)

    def search_related_goals(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for goals related to the query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of related goal information
        """
        self._stats["searches"] += 1

        try:
            # Use backend search functionality
            results = self._backend.search(query, limit)

            # Filter for goal-related entries (context: prefix or goal in metadata)
            goal_results = []
            for result in results:
                key = result.get("key", "")
                # Check if it's a goal context or has goal-related metadata
                if key.startswith("context:") or "goal" in result.get("metadata", {}):
                    goal_results.append(result)

            return goal_results[:limit]

        except Exception as e:
            print(f"Error searching related goals: {e}")
            return []

    def clear_goal_data(self, goal_id: str) -> bool:
        """
        Clear all data associated with a specific goal.

        Args:
            goal_id: ID of the goal to clear data for

        Returns:
            True if successful, False otherwise
        """
        try:
            success = True

            # Clear goal context - use correct key format
            context_key = f"context:{goal_id}"
            if not self._backend.delete(context_key):
                success = False

            # Clear execution history
            history_key = f"history:{goal_id}"
            if not self._backend.delete(history_key):
                success = False

            # Clear from cache
            if goal_id in self._cache:
                del self._cache[goal_id]
            if goal_id in self._cache_ttl:
                del self._cache_ttl[goal_id]

            return success

        except Exception as e:
            print(f"Error clearing goal data: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.

        Returns:
            Dictionary containing usage statistics
        """
        stats = self._stats.copy()
        stats["cache_size"] = len(self._cache)
        stats["backend_name"] = self._backend.name

        # Add total_entries if backend supports it
        try:
            if hasattr(self._backend, "_data"):
                stats["total_entries"] = len(self._backend._data)

                # Count context and history entries
                context_entries = sum(1 for key in self._backend._data.keys() if key.startswith("context:"))
                history_entries = sum(1 for key in self._backend._data.keys() if key.startswith("history:"))

                stats["context_entries"] = context_entries
                stats["history_entries"] = history_entries
            else:
                stats["total_entries"] = 0
                stats["context_entries"] = 0
                stats["history_entries"] = 0
        except:
            stats["total_entries"] = 0
            stats["context_entries"] = 0
            stats["history_entries"] = 0

        return stats

    def is_full(self) -> bool:
        """
        Check if the memory backend is full.

        Returns:
            True if the backend is full, False otherwise.
        """
        return self._backend.is_full()

    def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a result matches the given filters."""
        # Simple filter matching - could be enhanced
        return True  # For now, accept all results

    def get_performance_metrics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a node.

        Args:
            node_id: The node ID

        Returns:
            Performance metrics if found, None otherwise
        """
        return self.get_context(node_id, "performance")

    def get_domain_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get domain-specific context for a node.

        Args:
            node_id: The node ID

        Returns:
            Domain context if found, None otherwise
        """
        return self.get_context(node_id, "domain")

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        for key in self._stats:
            self._stats[key] = 0
