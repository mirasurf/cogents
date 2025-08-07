"""
Simple test script for Weaviate integration.

This script provides a quick way to test if the Weaviate client is working correctly.
Run this before running the full example to ensure your setup is correct.
"""

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cogents.common.semsearch.weaviate_client import DocumentChunk, WeaviateConfig, WeaviateManager


@pytest.mark.integration
@pytest.mark.slow
class TestWeaviateIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup Weaviate connection."""
        self.config = WeaviateConfig(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
        )

    def test_weaviate_connection(self):
        """Test basic Weaviate connection with embeddings (requires Ollama)."""
        print("Testing Weaviate connection...")

        manager = WeaviateManager(config=self.config)
        stored_ids = []

        try:
            assert manager.connect(), "Failed to connect to Weaviate"
            print("✓ Successfully connected to Weaviate!")

            # Test basic stats retrieval
            stats = manager.get_collection_stats()
            print(f"✓ Collection stats retrieved: {stats}")
            assert isinstance(stats, dict), "Stats should be a dictionary"

            # Store a test chunk
            test_chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=(
                    "This is a test document chunk for integration testing. "
                    "It contains information about machine learning and artificial intelligence. "
                    "The document discusses various algorithms and techniques used in modern AI systems."
                ),
                source_url="test://example.com",
                source_title="Test Document - AI and ML",
                chunk_index=0,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "test": True,
                    "category": "technology",
                    "topics": ["AI", "ML", "algorithms"],
                },
            )

            stored_ids = manager.store_chunks([test_chunk])
            print(f"✓ Successfully stored test chunk: {stored_ids}")
            assert len(stored_ids) > 0, "Should have stored at least one chunk"

            # Wait for indexing
            print("⏳ Waiting for document to be indexed...")
            time.sleep(6)

            # Verify object was stored by checking collection stats
            updated_stats = manager.get_collection_stats()
            print(f"✓ Updated collection stats: {updated_stats}")
            assert updated_stats.get("total_chunks", 0) > 0, "Should have at least one chunk in collection"

            # Perform semantic search
            query = "machine learning and AI algorithms"
            results = manager.search(query, limit=5, min_score=0.0)
            print(f"✓ Search returned {len(results)} results for query: '{query}'")

            assert len(results) > 0, "Should have found at least one result"
            print("✓ Found stored document via search")

            # Cleanup
            deleted_count = manager.delete_chunks(stored_ids)
            print("✓ Test chunk deleted")
            assert deleted_count > 0, "Should have deleted at least one chunk"

            manager.close()
            print("✓ Connection closed")

        except Exception as e:
            print(f"✗ Error during testing: {e}")
            if len(stored_ids) > 0:
                manager.delete_chunks(stored_ids)
            raise
