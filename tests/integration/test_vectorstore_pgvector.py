import uuid

import pytest

from cogents.capabilities.vectorstore.pgvector import PGVectorStore


@pytest.mark.integration
class TestPGVectorStoreIntegration:
    """Integration tests for PGVectorStore."""

    @pytest.fixture(autouse=True)
    def setup_vectorstore(self, test_collection_name, embedding_dims, pgvector_config):
        """Setup PGVector store for testing."""
        self.dbname = pgvector_config["dbname"]
        self.collection_name = test_collection_name
        self.embedding_dims = embedding_dims
        self.user = pgvector_config["user"]
        self.password = pgvector_config["password"]
        self.host = pgvector_config["host"]
        self.port = pgvector_config["port"]
        self.diskann = pgvector_config["diskann"]
        self.hnsw = pgvector_config["hnsw"]

        self.vectorstore = PGVectorStore(
            dbname=self.dbname,
            collection_name=self.collection_name,
            embedding_model_dims=self.embedding_dims,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            diskann=self.diskann,
            hnsw=self.hnsw,
        )

        yield

        # Cleanup
        try:
            self.vectorstore.delete_col()
            self.vectorstore.conn.close()
        except Exception:
            pass

    def test_create_collection(self):
        """Test collection creation."""
        # Collection should be created in setup
        cols = self.vectorstore.list_cols()
        assert self.collection_name in cols

    def test_insert_and_get(self):
        """Test inserting vectors and retrieving them."""
        test_vectors = [[0.1, 0.2, 0.3] * 256]  # 768 dimensions
        test_payloads = [{"data": "test object", "hash": "abc123"}]
        test_ids = [str(uuid.uuid4())]

        # Insert
        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # Get by ID
        result = self.vectorstore.get(test_ids[0])
        assert result is not None
        assert result.id == test_ids[0]
        assert result.payload["data"] == "test object"

    def test_search(self):
        """Test vector search functionality."""
        # Insert test data
        test_vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256, [0.7, 0.8, 0.9] * 256]
        test_payloads = [
            {"data": "first object", "category": "A"},
            {"data": "second object", "category": "B"},
            {"data": "third object", "category": "A"},
        ]
        test_ids = [str(uuid.uuid4()) for _ in range(3)]

        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # Search with query vector similar to first vector
        query_vector = [0.1, 0.2, 0.3] * 256
        results = self.vectorstore.search("test query", query_vector, limit=2)

        assert len(results) > 0
        assert results[0].score is not None

    def test_search_with_filters(self):
        """Test search with filters."""
        # Insert test data
        test_vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256]
        test_payloads = [
            {"data": "first object", "category": "A", "user_id": "user1"},
            {"data": "second object", "category": "B", "user_id": "user2"},
        ]
        test_ids = [str(uuid.uuid4()) for _ in range(2)]

        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # Search with filter
        query_vector = [0.1, 0.2, 0.3] * 256
        filters = {"user_id": "user1"}
        results = self.vectorstore.search("test query", query_vector, limit=5, filters=filters)

        assert len(results) > 0
        assert all(result.payload.get("user_id") == "user1" for result in results)

    def test_update(self):
        """Test updating vector and payload."""
        # Insert initial data
        test_vector = [0.1, 0.2, 0.3] * 256
        test_payload = {"data": "original", "version": 1}
        test_id = str(uuid.uuid4())

        self.vectorstore.insert([test_vector], [test_payload], [test_id])

        # Update payload
        new_payload = {"data": "updated", "version": 2}
        self.vectorstore.update(test_id, payload=new_payload)

        # Verify update
        result = self.vectorstore.get(test_id)
        assert result.payload["data"] == "updated"
        assert result.payload["version"] == 2

        # Update vector
        new_vector = [0.4, 0.5, 0.6] * 256
        self.vectorstore.update(test_id, vector=new_vector)

        # Verify vector update
        result = self.vectorstore.get(test_id)
        assert result.payload["data"] == "updated"  # Payload should remain

    def test_delete(self):
        """Test deleting vectors."""
        # Insert test data
        test_vector = [0.1, 0.2, 0.3] * 256
        test_payload = {"data": "to delete"}
        test_id = str(uuid.uuid4())

        self.vectorstore.insert([test_vector], [test_payload], [test_id])

        # Verify insertion
        result = self.vectorstore.get(test_id)
        assert result is not None

        # Delete
        self.vectorstore.delete(test_id)

        # Verify deletion
        result = self.vectorstore.get(test_id)
        assert result is None

    def test_list(self):
        """Test listing all vectors."""
        # Insert multiple vectors
        test_vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256]
        test_payloads = [{"data": "first", "category": "A"}, {"data": "second", "category": "B"}]
        test_ids = [str(uuid.uuid4()) for _ in range(2)]

        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # List all
        results = self.vectorstore.list()
        assert len(results) >= 2

        # List with filter
        filtered_results = self.vectorstore.list(filters={"category": "A"})
        assert len(filtered_results) > 0
        assert all(result.payload.get("category") == "A" for result in filtered_results)

    def test_col_info(self):
        """Test collection information retrieval."""
        info = self.vectorstore.col_info()
        assert isinstance(info, dict)
        assert "table_name" in info or "collection_name" in info

    def test_reset(self):
        """Test collection reset."""
        # Insert some data
        test_vector = [0.1, 0.2, 0.3] * 256
        test_payload = {"data": "test"}
        test_id = str(uuid.uuid4())

        self.vectorstore.insert([test_vector], [test_payload], [test_id])

        # Verify data exists
        result = self.vectorstore.get(test_id)
        assert result is not None

        # Reset
        self.vectorstore.reset()

        # Verify data is gone
        result = self.vectorstore.get(test_id)
        assert result is None

        # Verify collection still exists
        cols = self.vectorstore.list_cols()
        assert self.collection_name in cols

    def test_batch_operations(self):
        """Test batch insert operations."""
        # Insert multiple vectors in batch
        test_vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256, [0.7, 0.8, 0.9] * 256, [0.1, 0.1, 0.1] * 256]
        test_payloads = [{"data": f"object_{i}", "batch": "test"} for i in range(4)]
        test_ids = [str(uuid.uuid4()) for _ in range(4)]

        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # Verify all were inserted
        for test_id in test_ids:
            result = self.vectorstore.get(test_id)
            assert result is not None

        # Search to verify they're searchable
        query_vector = [0.1, 0.2, 0.3] * 256
        results = self.vectorstore.search("batch test", query_vector, limit=10)
        assert len(results) >= 4
