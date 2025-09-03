import uuid

import pytest

from cogents.base.base_vectorstore import BaseVectorStore
from cogents.ingreds.vectorstore.pgvector import PGVectorStore
from cogents.ingreds.vectorstore.weaviate import WeaviateVectorStore


@pytest.mark.integration
class TestVectorStoreBaseInterface:
    """Test BaseVectorStore interface compliance for all implementations."""

    @pytest.fixture(params=["weaviate", "pgvector"])
    def vectorstore(self, request, test_collection_name, embedding_dims, weaviate_config, pgvector_config):
        """Create vectorstore instance based on parameter."""
        if request.param == "weaviate":
            store = WeaviateVectorStore(
                collection_name=test_collection_name,
                embedding_model_dims=embedding_dims,
                cluster_url=weaviate_config["cluster_url"],
            )
        elif request.param == "pgvector":
            store = PGVectorStore(
                dbname=pgvector_config["dbname"],
                collection_name=test_collection_name,
                embedding_model_dims=embedding_dims,
                user=pgvector_config["user"],
                password=pgvector_config["password"],
                host=pgvector_config["host"],
                port=pgvector_config["port"],
                diskann=pgvector_config["diskann"],
                hnsw=pgvector_config["hnsw"],
            )

        yield store

        # Cleanup
        try:
            store.delete_col()
            if hasattr(store, "conn"):
                store.conn.close()
        except Exception:
            pass

    def test_interface_compliance(self, vectorstore):
        """Test that vectorstore implements all required methods."""
        assert isinstance(vectorstore, BaseVectorStore)

        # Test all abstract methods exist
        required_methods = [
            "create_col",
            "insert",
            "search",
            "delete",
            "update",
            "get",
            "list_cols",
            "delete_col",
            "col_info",
            "list",
            "reset",
        ]

        for method_name in required_methods:
            assert hasattr(vectorstore, method_name), f"Missing method: {method_name}"
            assert callable(getattr(vectorstore, method_name)), f"Method not callable: {method_name}"

    def test_basic_crud_operations(self, vectorstore):
        """Test basic CRUD operations work consistently."""
        # Create
        test_vector = [0.1, 0.2, 0.3] * 256
        test_payload = {"data": "test", "category": "test"}
        test_id = str(uuid.uuid4())

        vectorstore.insert([test_vector], [test_payload], [test_id])

        # Read
        result = vectorstore.get(test_id)
        assert result is not None
        assert result.id == test_id
        assert result.payload["data"] == "test"

        # Update
        new_payload = {"data": "updated", "category": "test"}
        vectorstore.update(test_id, payload=new_payload)

        updated_result = vectorstore.get(test_id)
        assert updated_result.payload["data"] == "updated"

        # Delete
        vectorstore.delete(test_id)
        deleted_result = vectorstore.get(test_id)
        assert deleted_result is None

    def test_search_consistency(self, vectorstore):
        """Test search behavior is consistent."""
        # Insert test data
        vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256, [0.7, 0.8, 0.9] * 256]
        payloads = [
            {"data": "first", "category": "A"},
            {"data": "second", "category": "B"},
            {"data": "third", "category": "A"},
        ]
        ids = [str(uuid.uuid4()) for _ in range(3)]

        vectorstore.insert(vectors, payloads, ids)

        # Test search returns results
        query_vector = [0.1, 0.2, 0.3] * 256
        results = vectorstore.search("test query", query_vector, limit=5)

        assert len(results) > 0
        assert all(hasattr(result, "id") for result in results)
        assert all(hasattr(result, "score") for result in results)
        assert all(hasattr(result, "payload") for result in results)

    def test_list_operations(self, vectorstore):
        """Test list operations work consistently."""
        # Insert test data
        vectors = [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256]
        payloads = [{"data": "first", "category": "A"}, {"data": "second", "category": "B"}]
        ids = [str(uuid.uuid4()) for _ in range(2)]

        vectorstore.insert(vectors, payloads, ids)

        # Test list all
        all_results = vectorstore.list()
        assert len(all_results) >= 2

        # Test list with filters
        filtered_results = vectorstore.list(filters={"category": "A"})
        assert len(filtered_results) > 0
        assert all(result.payload.get("category") == "A" for result in filtered_results)

    def test_collection_management(self, vectorstore):
        """Test collection management operations."""
        # Test collection info
        info = vectorstore.col_info()
        assert isinstance(info, dict)

        # Test list collections
        cols = vectorstore.list_cols()
        assert isinstance(cols, list)
        assert vectorstore.collection_name in cols

    def test_reset_functionality(self, vectorstore):
        """Test reset functionality."""
        # Insert data
        test_vector = [0.1, 0.2, 0.3] * 256
        test_payload = {"data": "test"}
        test_id = str(uuid.uuid4())

        vectorstore.insert([test_vector], [test_payload], [test_id])

        # Verify data exists
        assert vectorstore.get(test_id) is not None

        # Reset
        vectorstore.reset()

        # Verify data is gone but collection exists
        assert vectorstore.get(test_id) is None
        assert vectorstore.collection_name in vectorstore.list_cols()
