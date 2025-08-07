# Semantic Search Module

The semantic search module provides intelligent document retrieval capabilities by combining local vector search with web search fallback. It uses Weaviate as the vector database and Ollama for embeddings.

## Features

- **Hybrid Search**: Combines local vector search with web search fallback
- **Document Chunking**: Intelligent text splitting using LangChain
- **Vector Storage**: Weaviate integration with Ollama embeddings
- **Caching**: Query result caching for performance
- **Flexible Configuration**: Configurable chunking, search behavior, and connections
- **Multiple Search Engines**: Extensible web search interface (currently supports Tavily)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ SemanticSearch   │───▶│ Search Results  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌─────────────┐ ┌─────────┐ ┌──────────────┐
            │ Weaviate    │ │ Web     │ │ Document     │
            │ Manager     │ │ Search  │ │ Processor    │
            └─────────────┘ └─────────┘ └──────────────┘
                    │           │           │
                    ▼           ▼           ▼
            ┌─────────────┐ ┌─────────┐ ┌──────────────┐
            │ Vector DB   │ │ Tavily  │ │ LangChain    │
            │ (Weaviate)  │ │ API     │ │ Splitters    │
            └─────────────┘ └─────────┘ └──────────────┘
                    │                           │
                    ▼                           ▼
            ┌─────────────┐           ┌──────────────┐
            │ Ollama      │           │ Text Chunks  │
            │ Embeddings  │           │              │
            └─────────────┘           └──────────────┘
```

## Quick Start

### Prerequisites

1. **Weaviate**: Running on localhost:8080
2. **Ollama**: Running on localhost:11434 with `nomic-embed-text` model
3. **Environment**: `TAVILY_API_KEY` set for web search

### Basic Usage

```python
from cogents.common.semsearch import SemanticSearch

# Initialize semantic search
search_system = SemanticSearch()

# Connect to services
if search_system.connect():
    # Perform search
    results = search_system.search("travel destinations in Japan")
    
    # Print results
    for chunk, score in results.chunks:
        print(f"Score: {score:.3f}")
        print(f"Title: {chunk.source_title}")
        print(f"Content: {chunk.content[:200]}...")
        print("---")
    
    # Cleanup
    search_system.close()
```

### Advanced Configuration

```python
from cogents.common.semsearch import (
    SemanticSearch, SemanticSearchConfig, 
    WeaviateConfig, ChunkingConfig
)

# Configure Weaviate
weaviate_config = WeaviateConfig(
    host="localhost",
    port=8080,
    collection_name="MyDocuments",
    embedding_model="nomic-embed-text"
)

# Configure chunking
chunking_config = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)

# Configure search behavior
search_config = SemanticSearchConfig(
    weaviate_config=weaviate_config,
    chunking_config=chunking_config,
    local_search_limit=15,
    min_local_score=0.8,
    fallback_threshold=5,
    enable_caching=True,
    auto_store_web_results=True
)

# Create system
search_system = SemanticSearch(config=search_config)
```

## Workflow

### 1. Search Process

When you perform a search, the system follows this workflow:

1. **Cache Check**: Checks for cached results
2. **Local Search**: Searches Weaviate vector database
3. **Fallback Decision**: Determines if web search is needed based on:
   - Number of local results vs. fallback threshold
   - Force web search flag
4. **Web Search**: If needed, fetches results from web (Tavily)
5. **Document Processing**: Chunks web results using LangChain
6. **Storage**: Optionally stores new chunks in Weaviate
7. **Result Merging**: Combines and deduplicates results
8. **Caching**: Caches final results

### 2. Document Storage

Documents are processed and stored as follows:

1. **Text Splitting**: Uses `RecursiveCharacterTextSplitter`
2. **Chunk Creation**: Creates `DocumentChunk` objects with metadata
3. **Embedding**: Weaviate generates embeddings via Ollama
4. **Storage**: Stores in Weaviate collection

## Configuration

### Weaviate Configuration

```python
WeaviateConfig(
    host="localhost",              # Weaviate host
    port=8080,                     # Weaviate port
    grpc_port=50051,              # gRPC port
    use_ssl=False,                # SSL connection
    ollama_host="localhost",       # Ollama host
    ollama_port=11434,            # Ollama port
    collection_name="Documents",   # Collection name
    embedding_model="nomic-embed-text"  # Embedding model
)
```

### Chunking Configuration

```python
ChunkingConfig(
    chunk_size=1000,              # Target chunk size
    chunk_overlap=200,            # Overlap between chunks
    separators=["\n\n", "\n", ". "],  # Text separators
    length_function="len",        # Length measurement
    is_separator_regex=False      # Regex separators
)
```

### Search Configuration

```python
SemanticSearchConfig(
    local_search_limit=10,        # Max local results
    min_local_score=0.7,         # Min similarity score
    web_search_limit=5,          # Max web results
    fallback_threshold=3,        # Min local before web
    enable_caching=True,         # Enable caching
    cache_ttl_hours=24,          # Cache TTL
    auto_store_web_results=True, # Store web results
    merge_small_chunks=True      # Optimize chunks
)
```

## API Reference

### SemanticSearch

Main orchestrator class for semantic search operations.

#### Methods

- `connect() -> bool`: Connect to Weaviate
- `search(query, limit=None, force_web_search=False, filters=None) -> SemanticSearchResult`: Perform search
- `store_document(content, source_url, source_title, metadata=None) -> int`: Store document manually
- `get_stats() -> Dict[str, Any]`: Get system statistics
- `clear_cache()`: Clear query cache
- `close()`: Close connections

### WeaviateManager

Manages Weaviate database operations.

#### Methods

- `connect() -> bool`: Connect to Weaviate
- `store_chunks(chunks) -> List[str]`: Store document chunks
- `search(query, limit, min_score, filters=None) -> List[Tuple[DocumentChunk, float]]`: Vector search
- `delete_chunks(chunk_ids) -> int`: Delete chunks
- `get_collection_stats() -> Dict[str, Any]`: Get collection statistics

### DocumentProcessor

Handles document chunking and processing.

#### Methods

- `process_search_response(search_response) -> List[ProcessedDocument]`: Process search results
- `process_search_result(search_result) -> ProcessedDocument`: Process single result
- `process_raw_text(text, source_url, source_title, metadata=None) -> ProcessedDocument`: Process raw text
- `merge_chunks(chunks, max_size=None) -> List[DocumentChunk]`: Optimize chunks

## Error Handling

The module defines custom exceptions:

- `SemanticSearchError`: General semantic search errors
- `WeaviateError`: Weaviate operation errors
- `TavilySearchError`: Web search errors

## Performance Considerations

### Optimization Tips

1. **Chunk Size**: Balance between context and retrieval granularity
2. **Overlap**: Ensure important information isn't split
3. **Caching**: Enable for repeated queries
4. **Filtering**: Use metadata filters to narrow search
5. **Batch Operations**: Store multiple documents together

### Monitoring

```python
# Get system statistics
stats = search_system.get_stats()
print(f"Total chunks: {stats['weaviate']['total_chunks']}")
print(f"Cache size: {stats['cache_size']}")
print(f"Connected: {stats['connected']}")
```

## Examples

See `examples/semantic_search_example.py` for comprehensive usage examples including:

- Basic search operations
- Document storage
- Filtered searches
- Configuration options
- Error handling
- Performance monitoring

## Dependencies

- `weaviate-client`: Vector database client
- `langchain-text-splitters`: Document chunking
- `langchain-tavily`: Web search integration
- `pydantic`: Data validation and configuration

## Environment Variables

- `TAVILY_API_KEY`: Required for web search functionality

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure Weaviate and Ollama are running
2. **No Embeddings**: Check Ollama model is installed (`ollama pull nomic-embed-text`)
3. **Web Search Failed**: Verify `TAVILY_API_KEY` is set
4. **Memory Issues**: Reduce `chunk_size` or `local_search_limit`

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about search operations, connections, and data processing.