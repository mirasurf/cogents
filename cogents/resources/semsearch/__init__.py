"""
Semantic Search module for cogents.

This module provides semantic search capabilities using Weaviate as the vector database
and combining web search results for comprehensive information retrieval.
"""

from .document_processor import DocumentProcessor
from .semantic_search import SemanticSearch, SemanticSearchConfig
from .weaviate_client import WeaviateManager

__all__ = [
    "SemanticSearch",
    "WeaviateManager",
    "DocumentProcessor",
    "SemanticSearchConfig",
]
