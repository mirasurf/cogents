"""
Unified test configuration for cogents.

This file provides shared fixtures and configuration for both unit and integration tests.
"""
import os
import uuid

import pytest

# Import common test fixtures from existing modules
try:
    from tests.unit.goalith.conftest import (
        decomposer_registry,
        empty_graph_store,
        memory_store,
        mock_context,
        populated_graph_store,
        sample_goal_node,
        sample_subgoal_node,
        sample_task_node,
        sample_update_event,
    )
except ImportError:
    # If goalith tests aren't available, create empty fixtures
    @pytest.fixture
    def sample_goal_node():
        return None

    @pytest.fixture
    def sample_task_node():
        return None

    @pytest.fixture
    def sample_subgoal_node():
        return None

    @pytest.fixture
    def empty_graph_store():
        return None

    @pytest.fixture
    def populated_graph_store():
        return None

    @pytest.fixture
    def sample_update_event():
        return None

    @pytest.fixture
    def memory_store():
        return None

    @pytest.fixture
    def decomposer_registry():
        return None

    @pytest.fixture
    def mock_context():
        return {}


# Vectorstore integration test fixtures
@pytest.fixture(scope="session")
def test_collection_name():
    """Generate a unique collection name for testing."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def embedding_dims():
    """Standard embedding dimensions for testing."""
    return 768


@pytest.fixture(scope="session")
def weaviate_config():
    """Weaviate connection configuration."""
    return {
        "cluster_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        "auth_client_secret": os.getenv("WEAVIATE_AUTH_SECRET"),
        "additional_headers": {},
    }


@pytest.fixture(scope="session")
def pgvector_config():
    """PGVector connection configuration."""
    return {
        "dbname": os.getenv("PGVECTOR_DB", "test_vectorstore"),
        "user": os.getenv("PGVECTOR_USER", "postgres"),
        "password": os.getenv("PGVECTOR_PASSWORD", "postgres"),
        "host": os.getenv("PGVECTOR_HOST", "localhost"),
        "port": int(os.getenv("PGVECTOR_PORT", "5432")),
        "diskann": os.getenv("PGVECTOR_DISKANN", "false").lower() == "true",
        "hnsw": os.getenv("PGVECTOR_HNSW", "true").lower() == "true",
    }


# LLM integration test fixtures
@pytest.fixture(scope="session")
def openai_config():
    """OpenAI configuration for integration tests."""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    }


@pytest.fixture(scope="session")
def ollama_config():
    """Ollama configuration for integration tests."""
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama2"),
    }


@pytest.fixture(scope="session")
def llama_cpp_config():
    """LlamaCPP configuration for integration tests."""
    return {
        "model_path": os.getenv("LLAMACPP_MODEL_PATH"),
        "n_ctx": int(os.getenv("LLAMACPP_N_CTX", "2048")),
        "n_gpu_layers": int(os.getenv("LLAMACPP_N_GPU_LAYERS", "0")),
    }


@pytest.fixture(scope="session")
def openrouter_config():
    """OpenRouter configuration for integration tests."""
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku"),
    }


# Web search integration test fixtures
@pytest.fixture(scope="session")
def tavily_config():
    """Tavily search configuration for integration tests."""
    return {"api_key": os.getenv("TAVILY_API_KEY")}


@pytest.fixture(scope="session")
def google_ai_search_config():
    """Google AI Search configuration for integration tests."""
    return {
        "api_key": os.getenv("GOOGLE_AI_SEARCH_API_KEY"),
        "search_engine_id": os.getenv("GOOGLE_AI_SEARCH_ENGINE_ID"),
    }


# Common test utilities
@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock response for testing purposes.",
        "model": "test-model",
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    }


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "content": "This is the first chunk of a test document.",
            "metadata": {"source": "test.txt", "chunk_index": 0},
        },
        {
            "id": str(uuid.uuid4()),
            "content": "This is the second chunk of a test document.",
            "metadata": {"source": "test.txt", "chunk_index": 1},
        },
    ]


@pytest.fixture
def test_vectors():
    """Sample vectors for testing."""
    return [[0.1, 0.2, 0.3] * 256, [0.4, 0.5, 0.6] * 256, [0.7, 0.8, 0.9] * 256]  # 768 dimensions


@pytest.fixture
def test_payloads():
    """Sample payloads for testing."""
    return [
        {"data": "first object", "category": "A"},
        {"data": "second object", "category": "B"},
        {"data": "third object", "category": "A"},
    ]


@pytest.fixture
def test_ids():
    """Sample IDs for testing."""
    return [str(uuid.uuid4()) for _ in range(3)]


# Environment and configuration helpers
@pytest.fixture(scope="session")
def test_environment():
    """Get test environment configuration."""
    return {
        "is_integration": os.getenv("INTEGRATION_TESTS", "false").lower() == "true",
        "is_slow": os.getenv("SLOW_TESTS", "false").lower() == "true",
        "debug": os.getenv("TEST_DEBUG", "false").lower() == "true",
    }


def pytest_configure(config):
    """Configure pytest with markers and options."""
    # Add custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "llm: mark test as requiring LLM")
    config.addinivalue_line("markers", "vectorstore: mark test as requiring vector store")
    config.addinivalue_line("markers", "websearch: mark test as requiring web search")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    # Skip integration tests if not requested
    if not config.getoption("--runintegration", default=False):
        skip_integration = pytest.mark.skip(reason="Integration tests not enabled")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip slow tests if not requested
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="Slow tests not enabled")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runintegration", action="store_true", default=False, help="run integration tests")
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--llm", action="store_true", default=False, help="run LLM-dependent tests")
    parser.addoption("--vectorstore", action="store_true", default=False, help="run vectorstore-dependent tests")
    parser.addoption("--websearch", action="store_true", default=False, help="run web search-dependent tests")
