import uuid

import pytest

from cogents.ingreds.vectorstore.weaviate import WeaviateVectorStore


@pytest.mark.integration
class TestWeaviateVectorStoreIntegration:
    """Integration tests for WeaviateVectorStore."""

    @pytest.fixture(autouse=True)
    def setup_vectorstore(self, test_collection_name, embedding_dims, weaviate_config):
        """Setup Weaviate vector store for testing."""
        self.collection_name = test_collection_name
        self.embedding_dims = embedding_dims
        self.cluster_url = weaviate_config["cluster_url"]

        self.vectorstore = WeaviateVectorStore(
            collection_name=self.collection_name, embedding_model_dims=self.embedding_dims, cluster_url=self.cluster_url
        )

        yield

        # Cleanup
        try:
            self.vectorstore.delete_col()
        except Exception:
            pass

    def test_create_collection(self):
        """Test collection creation."""
        # Collection should be created in setup
        cols = self.vectorstore.list_cols()
        assert self.collection_name in cols

    def test_insert_and_get(self, test_vectors, test_payloads, test_ids):
        """Test inserting vectors and retrieving them."""
        # Insert
        self.vectorstore.insert([test_vectors[0]], [test_payloads[0]], [test_ids[0]])

        # Get by ID
        result = self.vectorstore.get(test_ids[0])
        assert result is not None
        assert result.id == test_ids[0]
        assert result.payload["data"] == test_payloads[0]["data"]

    def test_search(self, test_vectors, test_payloads, test_ids):
        """Test vector search functionality."""
        # Insert test data
        self.vectorstore.insert(test_vectors, test_payloads, test_ids)

        # Search with query vector similar to first vector
        query_vector = test_vectors[0]
        results = self.vectorstore.search("test query", query_vector, limit=2)

        assert len(results) > 0
        assert results[0].score is not None

    def test_search_with_filters(self, test_vectors, test_payloads, test_ids):
        """Test search with filters."""
        # Insert test data with user_id
        payloads_with_user = [{**test_payloads[0], "user_id": "user1"}, {**test_payloads[1], "user_id": "user2"}]
        self.vectorstore.insert(test_vectors[:2], payloads_with_user, test_ids[:2])

        # Search with filter
        query_vector = test_vectors[0]
        filters = {"user_id": "user1"}
        results = self.vectorstore.search("test query", query_vector, limit=5, filters=filters)

        assert len(results) > 0
        assert all(result.payload.get("user_id") == "user1" for result in results)

    def test_update(self, test_vectors, test_payloads, test_ids):
        """Test updating vector and payload."""
        # Insert initial data
        test_payload = {**test_payloads[0], "version": 1}
        self.vectorstore.insert([test_vectors[0]], [test_payload], [test_ids[0]])

        # Update payload
        new_payload = {**test_payloads[0], "data": "updated", "version": 2}
        self.vectorstore.update(test_ids[0], payload=new_payload)

        # Verify update
        result = self.vectorstore.get(test_ids[0])
        assert result.payload["data"] == "updated"
        assert result.payload["version"] == 2

        # Update vector
        new_vector = test_vectors[1]
        self.vectorstore.update(test_ids[0], vector=new_vector)

        # Verify vector update
        result = self.vectorstore.get(test_ids[0])
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
        assert "name" in info or "collection_name" in info

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
