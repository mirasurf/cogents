"""
Weaviate client manager for semantic search.

This module provides a high-level interface for interacting with Weaviate,
including document storage, retrieval, and vector search capabilities.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import weaviate
from pydantic import BaseModel, Field
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from cogents.common.consts import OLLAMA_EMBEDDING_MODEL, OLLAMA_GENERATIVE_MODEL

from .types import DocumentChunk

logger = logging.getLogger(__name__)

# Suppress httpx logs
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


__all__ = [
    "WeaviateConfig",
    "WeaviateManager",
    "WeaviateError",
]


class WeaviateError(Exception):
    """Custom exception for Weaviate operations."""


class WeaviateConfig(BaseModel):
    """Configuration for Weaviate connection."""

    host: str = Field(default="localhost", description="Weaviate host")
    port: int = Field(default=8080, description="Weaviate port")
    grpc_port: int = Field(default=50051, description="Weaviate gRPC port")
    use_ssl: bool = Field(default=False, description="Use SSL connection")
    ollama_host: str = Field(default="localhost", description="Ollama host for embeddings")
    ollama_port: int = Field(default=11434, description="Ollama port")
    collection_name: str = Field(default="DocumentChunks", description="Weaviate collection name")
    embedding_model: str = Field(default=OLLAMA_EMBEDDING_MODEL, description="Ollama embedding model")
    generative_model: str = Field(default=OLLAMA_GENERATIVE_MODEL, description="Ollama generative model")


class WeaviateManager:
    """
    Manager class for Weaviate operations.

    Handles connection, collection management, document storage, and semantic search
    operations with Weaviate vector database.
    """

    def __init__(self, config: Optional[WeaviateConfig] = None):
        """
        Initialize Weaviate manager.

        Args:
            config: Weaviate configuration object
        """
        self.config = config or WeaviateConfig()
        self._client = None
        self._collection = None

    def connect(self) -> bool:
        """
        Connect to Weaviate instance.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Construct connection URL
            scheme = "https" if self.config.use_ssl else "http"
            url = f"{scheme}://{self.config.host}:{self.config.port}"

            self._client = weaviate.connect_to_custom(
                http_host=self.config.host,
                http_port=self.config.port,
                http_secure=self.config.use_ssl,
                grpc_host=self.config.host,
                grpc_port=self.config.grpc_port,
                grpc_secure=self.config.use_ssl,
            )

            # Test connection
            if self._client.is_ready():
                self._setup_collection()
                return True
            else:
                logger.error("Failed to connect to Weaviate")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            return False

    def _setup_collection(self) -> None:
        """Setup the document chunks collection with proper schema."""
        try:
            collection_name = self.config.collection_name

            if self._client.collections.exists(collection_name):
                self._collection = self._client.collections.get(collection_name)
                return

            properties = [
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="source_title", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="timestamp", data_type=DataType.DATE),
                Property(name="metadata", data_type=DataType.TEXT),
            ]

            # Correct usage with class-based config objects
            self._collection = self._client.collections.create(
                name=collection_name,
                generative_config=Configure.Generative.ollama(
                    api_endpoint=f"http://{self.config.ollama_host}:{self.config.ollama_port}",
                    model=self.config.generative_model,
                ),
                vector_config=Configure.Vectorizer.text2vec_ollama(
                    api_endpoint=f"http://{self.config.ollama_host}:{self.config.ollama_port}",
                    model=self.config.embedding_model,
                ),
                properties=properties,
            )

            logger.info(f"Created new collection: {collection_name}")

        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise WeaviateError(f"Failed to setup collection: {e}")

    def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Store document chunks in Weaviate.

        Args:
            chunks: List of document chunks to store

        Returns:
            List[str]: List of stored chunk IDs

        Raises:
            WeaviateError: If storage operation fails
        """
        if self._collection is None:
            raise WeaviateError("Not connected to Weaviate")

        try:
            stored_ids = []

            for chunk in chunks:
                # Prepare object data
                import json

                obj_data = {
                    "content": chunk.content,
                    "source_url": chunk.source_url,
                    "source_title": chunk.source_title,
                    "chunk_index": chunk.chunk_index,
                    "timestamp": chunk.timestamp,
                    "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
                }

                # Insert object
                result = self._collection.data.insert(
                    properties=obj_data,
                    uuid=uuid.UUID(chunk.chunk_id) if chunk.chunk_id else None,
                )

                stored_ids.append(str(result))

            logger.info(f"Stored {len(stored_ids)} chunks in Weaviate")
            return stored_ids

        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise WeaviateError(f"Failed to store chunks: {e}")

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform semantic search in Weaviate.

        Args:
            query: Search query string
            limit: Maximum number of results
            min_score: Minimum similarity score
            filters: Optional filters for search

        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples

        Raises:
            WeaviateError: If search operation fails
        """
        if self._collection is None:
            raise WeaviateError("Not connected to Weaviate")

        try:
            # Build search query
            search_query = self._collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["score"],
            )

            # Apply filters if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(Filter.by_property(key).equal(value))

                if filter_conditions:
                    search_query = search_query.where(
                        Filter.all_of(filter_conditions) if len(filter_conditions) > 1 else filter_conditions[0]
                    )

            # Execute search
            response = search_query
            results = []

            for obj in response.objects:
                score = obj.metadata.score if obj.metadata.score else 0.0

                # Filter by minimum score
                if score >= min_score:
                    # Parse metadata from JSON string
                    import json

                    metadata_str = obj.properties.get("metadata", "{}")
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else {}
                    except json.JSONDecodeError:
                        metadata = {}

                    chunk = DocumentChunk(
                        chunk_id=str(obj.uuid),
                        content=obj.properties.get("content", ""),
                        source_url=obj.properties.get("source_url"),
                        source_title=obj.properties.get("source_title"),
                        chunk_index=obj.properties.get("chunk_index", 0),
                        timestamp=obj.properties.get("timestamp", datetime.now(timezone.utc)),
                        metadata=metadata,
                    )

                    results.append((chunk, score))

            logger.debug(f"Found {len(results)} relevant chunks for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise WeaviateError(f"Search failed: {e}")

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            int: Number of deleted chunks

        Raises:
            WeaviateError: If deletion fails
        """
        if self._collection is None:
            raise WeaviateError("Not connected to Weaviate")

        try:
            deleted_count = 0

            for chunk_id in chunk_ids:
                try:
                    self._collection.data.delete_by_id(uuid.UUID(chunk_id))
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete chunk {chunk_id}: {e}")

            logger.info(f"Deleted {deleted_count} chunks from Weaviate")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise WeaviateError(f"Failed to delete chunks: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics
        """
        if self._collection is None:
            raise WeaviateError("Not connected to Weaviate")

        try:
            # Get total object count
            total_objects = self._collection.aggregate.over_all(total_count=True)

            return {
                "total_chunks": total_objects.total_count,
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model,
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close the Weaviate connection."""
        if self._client:
            self._client.close()
