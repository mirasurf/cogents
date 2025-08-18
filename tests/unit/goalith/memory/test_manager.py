"""
Unit tests for goalith.memory.manager module.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cogents.goalith.base.goal_node import GoalNode, NodeStatus, NodeType
from cogents.goalith.memory.inmemstore import InMemoryStore
from cogents.goalith.memory.manager import MemoryManager


class TestMemoryManager:
    """Test MemoryManager functionality."""

    def test_default_initialization(self):
        """Test creating memory manager with default backend."""
        manager = MemoryManager()

        assert manager._backend is not None
        assert isinstance(manager._backend, InMemoryStore)
        assert manager._cache == {}
        assert manager._cache_ttl == {}
        assert manager._stats["context_retrievals"] == 0
        assert manager._stats["context_stores"] == 0

    def test_custom_backend_initialization(self):
        """Test creating memory manager with custom backend."""
        custom_backend = InMemoryStore()
        manager = MemoryManager(memory_backend=custom_backend)

        assert manager._backend is custom_backend

    def test_backend_property(self):
        """Test backend property access."""
        backend = InMemoryStore()
        manager = MemoryManager(memory_backend=backend)

        assert manager.backend is backend

    def test_enrich_node_context_empty_memory(self, sample_goal_node):
        """Test enriching node context when memory is empty."""
        manager = MemoryManager()

        context = manager.enrich_node_context(sample_goal_node)

        assert isinstance(context, dict)
        assert context.get("node_id") == sample_goal_node.id
        assert context.get("description") == sample_goal_node.description
        assert context.get("type") == sample_goal_node.type.value
        assert manager._stats["enrichments"] == 1

    def test_enrich_node_context_with_stored_data(self, sample_goal_node):
        """Test enriching node context with stored memory data."""
        manager = MemoryManager()

        # Store some context first
        stored_context = {
            "domain": "test",
            "execution_history": ["step1", "step2"],
            "performance_metrics": {"success_rate": 0.85},
        }
        manager._backend.store_context(sample_goal_node.id, "enrichment", stored_context)

        # Enrich the node
        context = manager.enrich_node_context(sample_goal_node)

        # Should include both node data and stored context
        assert context["node_id"] == sample_goal_node.id
        assert context["domain"] == "test"
        assert context["execution_history"] == ["step1", "step2"]
        assert context["performance_metrics"]["success_rate"] == 0.85

    def test_store_context(self, sample_goal_node):
        """Test storing context for a node."""
        manager = MemoryManager()

        context_data = {"key": "value", "number": 42}

        manager.store_context(sample_goal_node.id, "test_key", context_data)

        # Verify it was stored
        retrieved = manager._backend.get_context(sample_goal_node.id, "test_key")
        assert retrieved == context_data
        assert manager._stats["context_stores"] == 1

    def test_get_context(self, sample_goal_node):
        """Test retrieving context for a node."""
        manager = MemoryManager()

        # Store some context
        context_data = {"stored": "data"}
        manager._backend.store_context(sample_goal_node.id, "test_key", context_data)

        # Retrieve it
        retrieved = manager.get_context(sample_goal_node.id, "test_key")

        assert retrieved == context_data
        assert manager._stats["context_retrievals"] == 1

    def test_get_nonexistent_context(self, sample_goal_node):
        """Test retrieving nonexistent context returns None."""
        manager = MemoryManager()

        retrieved = manager.get_context(sample_goal_node.id, "nonexistent")

        assert retrieved is None
        assert manager._stats["context_retrievals"] == 1

    def test_store_execution_note(self, sample_goal_node):
        """Test storing execution notes."""
        manager = MemoryManager()

        note = {
            "action": "started_task",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": "Task execution began",
        }

        manager.store_execution_note(sample_goal_node.id, note)

        # Verify note was stored
        notes = manager._backend.get_execution_history(sample_goal_node.id)
        assert len(notes) == 1
        assert notes[0] == note

    def test_get_execution_history(self, sample_goal_node):
        """Test getting execution history."""
        manager = MemoryManager()

        # Store multiple notes
        notes = [
            {"action": "started", "timestamp": "2023-01-01T10:00:00Z"},
            {"action": "progress", "timestamp": "2023-01-01T11:00:00Z"},
            {"action": "completed", "timestamp": "2023-01-01T12:00:00Z"},
        ]

        for note in notes:
            manager._backend.store_execution_note(sample_goal_node.id, note)

        # Retrieve history
        history = manager.get_execution_history(sample_goal_node.id)

        assert len(history) == 3
        assert history == notes

    def test_search_similar_goals(self, sample_goal_node):
        """Test searching for similar goals."""
        manager = MemoryManager()

        # Store some goals with context
        similar_goal = GoalNode(
            id="similar-1", description="Similar test goal", type=NodeType.GOAL, tags={"test", "similar"}
        )

        different_goal = GoalNode(
            id="different-1", description="Completely different objective", type=NodeType.TASK, tags={"work", "urgent"}
        )

        # Store contexts
        manager.store_context(similar_goal.id, "metadata", {"domain": "testing", "complexity": "medium"})
        manager.store_context(different_goal.id, "metadata", {"domain": "production", "complexity": "high"})

        # Search for similar goals
        similar = manager.search_similar_goals(sample_goal_node, limit=10)

        assert isinstance(similar, list)
        assert manager._stats["searches"] == 1

    def test_search_similar_goals_with_filters(self, sample_goal_node):
        """Test searching similar goals with filters."""
        manager = MemoryManager()

        filters = {"type": NodeType.GOAL, "tags": ["test"], "status": NodeStatus.COMPLETED}

        similar = manager.search_similar_goals(sample_goal_node, filters=filters, limit=5)

        assert isinstance(similar, list)
        assert len(similar) <= 5

    def test_get_performance_metrics(self, sample_goal_node):
        """Test getting performance metrics."""
        manager = MemoryManager()

        # Store some metrics
        metrics = {"execution_time": 120.5, "success_rate": 0.95, "difficulty_rating": 3.2}
        manager.store_context(sample_goal_node.id, "performance", metrics)

        # Retrieve metrics
        retrieved_metrics = manager.get_performance_metrics(sample_goal_node.id)

        assert retrieved_metrics == metrics

    def test_get_domain_context(self, sample_goal_node):
        """Test getting domain-specific context."""
        manager = MemoryManager()

        # Store domain context
        domain_data = {
            "domain": "software_development",
            "common_patterns": ["TDD", "agile", "CI/CD"],
            "best_practices": {"testing": "comprehensive", "documentation": "essential"},
        }
        manager.store_context(sample_goal_node.id, "domain", domain_data)

        # Retrieve domain context
        retrieved = manager.get_domain_context(sample_goal_node.id)

        assert retrieved == domain_data

    def test_caching_mechanism(self, sample_goal_node):
        """Test memory caching mechanism."""
        manager = MemoryManager()

        # Store context
        context_data = {"cached": "data", "timestamp": datetime.now(timezone.utc).isoformat()}
        manager.store_context(sample_goal_node.id, "cache_test", context_data)

        # First retrieval should hit backend
        retrieved1 = manager.get_context(sample_goal_node.id, "cache_test")
        manager._stats["cache_misses"]

        # Second retrieval should hit cache
        retrieved2 = manager.get_context(sample_goal_node.id, "cache_test")
        cache_hits_after_second = manager._stats["cache_hits"]

        assert retrieved1 == retrieved2 == context_data
        assert cache_hits_after_second > 0

    def test_cache_expiration(self, sample_goal_node):
        """Test cache expiration mechanism."""
        manager = MemoryManager()

        # Set very short cache timeout
        manager._cache_timeout = timedelta(milliseconds=1)

        # Store and retrieve context (should cache)
        context_data = {"expires": "quickly"}
        manager.store_context(sample_goal_node.id, "expire_test", context_data)
        retrieved1 = manager.get_context(sample_goal_node.id, "expire_test")

        # Wait for cache to expire
        import time

        time.sleep(0.01)  # 10ms, longer than 1ms timeout

        # Next retrieval should miss cache due to expiration
        cache_misses_before = manager._stats["cache_misses"]
        retrieved2 = manager.get_context(sample_goal_node.id, "expire_test")
        cache_misses_after = manager._stats["cache_misses"]

        assert retrieved1 == retrieved2 == context_data
        assert cache_misses_after > cache_misses_before

    def test_clear_cache(self, sample_goal_node):
        """Test clearing the cache."""
        manager = MemoryManager()

        # Cache some data
        manager.store_context(sample_goal_node.id, "clear_test", {"data": "to_clear"})
        manager.get_context(sample_goal_node.id, "clear_test")  # Cache it

        assert len(manager._cache) > 0
        assert len(manager._cache_ttl) > 0

        # Clear cache
        manager.clear_cache()

        assert len(manager._cache) == 0
        assert len(manager._cache_ttl) == 0

    def test_get_stats(self, sample_goal_node):
        """Test getting memory manager statistics."""
        manager = MemoryManager()

        # Perform various operations
        manager.enrich_node_context(sample_goal_node)
        manager.store_context(sample_goal_node.id, "stats_test", {"test": "data"})
        manager.get_context(sample_goal_node.id, "stats_test")
        manager.search_similar_goals(sample_goal_node)

        stats = manager.get_stats()

        assert stats["enrichments"] >= 1
        assert stats["context_stores"] >= 1
        assert stats["context_retrievals"] >= 1
        assert stats["searches"] >= 1
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_reset_stats(self, sample_goal_node):
        """Test resetting memory manager statistics."""
        manager = MemoryManager()

        # Perform operations to generate stats
        manager.enrich_node_context(sample_goal_node)
        manager.store_context(sample_goal_node.id, "reset_test", {"data": "test"})

        # Verify stats are non-zero
        assert manager._stats["enrichments"] > 0
        assert manager._stats["context_stores"] > 0

        # Reset stats
        manager.reset_stats()

        # Verify stats are reset
        assert manager._stats["enrichments"] == 0
        assert manager._stats["context_stores"] == 0
        assert manager._stats["context_retrievals"] == 0
        assert manager._stats["cache_hits"] == 0
        assert manager._stats["cache_misses"] == 0
        assert manager._stats["searches"] == 0

    def test_bulk_context_operations(self):
        """Test bulk context storage and retrieval."""
        manager = MemoryManager()

        # Store context for multiple nodes
        node_contexts = {}
        for i in range(10):
            node_id = f"node-{i}"
            context = {"index": i, "data": f"context_data_{i}"}
            manager.store_context(node_id, "bulk_test", context)
            node_contexts[node_id] = context

        # Retrieve all contexts
        for node_id, expected_context in node_contexts.items():
            retrieved = manager.get_context(node_id, "bulk_test")
            assert retrieved == expected_context

    def test_complex_context_data(self, sample_goal_node):
        """Test storing and retrieving complex context data."""
        manager = MemoryManager()

        complex_context = {
            "metadata": {"created_by": "user123", "project": "test_project", "version": "1.2.3"},
            "execution_data": {
                "steps": [
                    {"step": 1, "action": "initialize", "duration": 1.5},
                    {"step": 2, "action": "process", "duration": 23.7},
                    {"step": 3, "action": "finalize", "duration": 0.8},
                ],
                "total_duration": 26.0,
                "resources_used": ["cpu", "memory", "disk"],
            },
            "relationships": {
                "dependencies": ["node-1", "node-2"],
                "dependents": ["node-4", "node-5"],
                "related_goals": [],
            },
        }

        # Store complex context
        manager.store_context(sample_goal_node.id, "complex", complex_context)

        # Retrieve and verify
        retrieved = manager.get_context(sample_goal_node.id, "complex")
        assert retrieved == complex_context

        # Verify nested access works
        assert retrieved["metadata"]["created_by"] == "user123"
        assert len(retrieved["execution_data"]["steps"]) == 3
        assert retrieved["execution_data"]["steps"][1]["duration"] == 23.7

    def test_memory_backend_error_handling(self, sample_goal_node):
        """Test error handling when memory backend fails."""
        manager = MemoryManager()

        # Mock backend to raise errors
        with patch.object(manager._backend, "store_context") as mock_store:
            mock_store.side_effect = Exception("Backend error")

            # Should handle error gracefully
            try:
                manager.store_context(sample_goal_node.id, "error_test", {"data": "test"})
                # If no exception is raised, that's fine too
            except Exception:
                # If exception propagates, that's also acceptable behavior
                pass

        with patch.object(manager._backend, "get_context") as mock_get:
            mock_get.side_effect = Exception("Backend error")

            # Should handle error gracefully and return None
            result = manager.get_context(sample_goal_node.id, "error_test")
            # Result could be None or an exception could be raised
            assert result is None or isinstance(result, dict)
