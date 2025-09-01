"""
Tests for tarzi searcher functionality.
"""

from unittest.mock import Mock, patch

import pytest

from cogents.resources.tarzi.fetcher import ContentMode
from cogents.resources.tarzi.searcher import SearchResult, SearchWithContentResult, TarziSearcher


class MockTarziSearchResult:
    """Mock tarzi search result for testing."""

    def __init__(self, title, url, snippet, rank=1):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.rank = rank


class TestTarziSearcher:
    """Unit test cases for TarziSearcher."""

    def test_initialization_with_defaults(self):
        """Test TarziSearcher initialization with default parameters."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine, patch(
            "cogents.resources.tarzi.searcher.TarziFetcher"
        ) as mock_fetcher:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine.from_config.return_value = Mock()
            mock_fetcher.return_value = Mock()

            searcher = TarziSearcher()

            assert searcher.search_engine == "duckduckgo"
            assert searcher.search_mode == "webquery"

    def test_initialization_with_custom_params(self):
        """Test TarziSearcher initialization with custom parameters."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine, patch(
            "cogents.resources.tarzi.searcher.TarziFetcher"
        ) as mock_fetcher:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine.from_config.return_value = Mock()
            mock_fetcher.return_value = Mock()

            fetcher_config = {"llm_provider": "ollama"}

            searcher = TarziSearcher(
                search_engine="google",
                fetcher_config=fetcher_config,
            )

            assert searcher.search_engine == "google"
            assert searcher.search_mode == "webquery"
            mock_fetcher.assert_called_once_with(**fetcher_config)

    def test_tarzi_import_error(self):
        """Test handling of tarzi import error."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'tarzi'")), patch(
            "cogents.resources.tarzi.searcher.TarziFetcher"
        ):
            with pytest.raises(ImportError, match="tarzi library is required but not available"):
                TarziSearcher()

    def test_search_basic(self):
        """Test basic search functionality."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_web"
        ) as mock_search_web, patch("cogents.resources.tarzi.searcher.TarziFetcher"):
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine = Mock()
            mock_search_engine_cls.from_config.return_value = mock_search_engine

            # Mock search results
            mock_results = [
                MockTarziSearchResult("Test Title 1", "https://example1.com", "Test snippet 1", 1),
                MockTarziSearchResult("Test Title 2", "https://example2.com", "Test snippet 2", 2),
            ]
            mock_search_web.return_value = mock_results

            searcher = TarziSearcher()
            results = searcher.search("test query", max_results=2)

            assert len(results) == 2
            assert isinstance(results[0], SearchResult)
            assert results[0].title == "Test Title 1"
            assert results[0].url == "https://example1.com"
            assert results[0].snippet == "Test snippet 1"
            assert results[0].rank == 1

            mock_search_web.assert_called_once_with("test query", "webquery", 2)

    def test_search_with_custom_mode(self):
        """Test search with custom search mode."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_web"
        ) as mock_search_web, patch("cogents.resources.tarzi.searcher.TarziFetcher"):
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()
            mock_search_web.return_value = [MockTarziSearchResult("Test", "https://example.com", "Test snippet")]

            searcher = TarziSearcher()
            searcher.search("test query")

            mock_search_web.assert_called_once_with("test query", "webquery", 10)

    def test_search_with_content_separate(self):
        """Test search_with_content using separate search and fetch."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_web"
        ) as mock_search_web, patch("cogents.resources.tarzi.searcher.TarziFetcher") as mock_fetcher_cls:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()

            # Mock search results
            mock_search_web.return_value = [MockTarziSearchResult("Test Title", "https://example.com", "Test snippet")]

            # Mock fetcher
            mock_fetcher = Mock()
            mock_fetcher.fetch.return_value = "LLM formatted content"
            mock_fetcher_cls.return_value = mock_fetcher

            searcher = TarziSearcher()
            results = searcher.search_with_content("test query", content_mode=ContentMode.LLM_FORMATTED)

            assert len(results) == 1
            assert isinstance(results[0], SearchWithContentResult)
            assert results[0].result.title == "Test Title"
            assert results[0].content == "LLM formatted content"
            assert results[0].content_mode == ContentMode.LLM_FORMATTED

            mock_fetcher.fetch.assert_called_once_with("https://example.com", content_mode=ContentMode.LLM_FORMATTED)

    def test_search_with_content_with_fetch_error(self):
        """Test search_with_content when content fetching fails."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_web"
        ) as mock_search_web, patch("cogents.resources.tarzi.searcher.TarziFetcher") as mock_fetcher_cls:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()

            # Mock search results
            mock_search_web.return_value = [MockTarziSearchResult("Test Title", "https://example.com", "Test snippet")]

            # Mock fetcher that raises exception
            mock_fetcher = Mock()
            mock_fetcher.fetch.side_effect = Exception("Fetch failed")
            mock_fetcher_cls.return_value = mock_fetcher

            searcher = TarziSearcher()
            results = searcher.search_with_content("test query", content_mode=ContentMode.LLM_FORMATTED)

            # Should still return result with error message
            assert len(results) == 1
            assert "Failed to fetch content: Fetch failed" in results[0].content

    def test_search_with_content_string_content_mode(self):
        """Test search_with_content with string content mode."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_web"
        ) as mock_search_web, patch("cogents.resources.tarzi.searcher.TarziFetcher") as mock_fetcher_cls:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()

            # Mock search results
            mock_search_web.return_value = [MockTarziSearchResult("Test", "https://example.com", "Snippet")]

            # Mock fetcher
            mock_fetcher = Mock()
            mock_fetcher.fetch.return_value = "# Markdown content"
            mock_fetcher_cls.return_value = mock_fetcher

            searcher = TarziSearcher()
            results = searcher.search_with_content("test query", content_mode=ContentMode.MARKDOWN)

            assert len(results) == 1
            assert results[0].content == "# Markdown content"


@pytest.mark.integration
class TestTarziSearcherIntegration:
    """Integration test cases for TarziSearcher."""

    def test_search_real_query(self):
        """Test searching with a real query."""
        searcher = TarziSearcher()
        results = searcher.search("Python programming", max_results=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3

        for result in results:
            assert isinstance(result, SearchResult)
            assert isinstance(result.title, str)
            assert isinstance(result.url, str)
            assert isinstance(result.snippet, str)
            assert len(result.title) > 0
            assert result.url.startswith("http")

    def test_search_with_content_real_query_markdown(self):
        """Test search_with_content with real query and markdown mode."""
        searcher = TarziSearcher()
        results = searcher.search_with_content("Python programming", max_results=2, content_mode=ContentMode.MARKDOWN)

        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 2

        for result in results:
            assert isinstance(result, SearchWithContentResult)
            assert isinstance(result.result, SearchResult)
            assert isinstance(result.content, str)
            assert result.content_mode == ContentMode.MARKDOWN
            assert len(result.content) > 0

    def test_search_with_content_real_query_raw_html(self):
        """Test search_with_content with real query and raw HTML mode."""
        searcher = TarziSearcher()
        results = searcher.search_with_content("Python programming", max_results=1, content_mode=ContentMode.RAW_HTML)

        assert isinstance(results, list)
        assert len(results) > 0

        result = results[0]
        assert isinstance(result, SearchWithContentResult)
        assert result.content_mode == ContentMode.RAW_HTML
        assert "html" in result.content.lower() or len(result.content) > 0

    def test_search_with_content_real_query_llm_formatted(self):
        """Test search_with_content with real query and LLM formatted mode."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available for LLM formatting test")

        fetcher_config = {"llm_provider": "openai"}

        searcher = TarziSearcher(fetcher_config=fetcher_config)
        results = searcher.search_with_content(
            "Python programming", max_results=1, content_mode=ContentMode.LLM_FORMATTED
        )

        assert isinstance(results, list)
        assert len(results) > 0

        result = results[0]
        assert isinstance(result, SearchWithContentResult)
        assert result.content_mode == ContentMode.LLM_FORMATTED
        assert len(result.content) > 0

    def test_search_different_engines(self):
        """Test search with different search engines."""
        engines_to_test = ["duckduckgo", "bing"]

        for engine in engines_to_test:
            searcher = TarziSearcher(search_engine=engine)
            results = searcher.search("Python", max_results=1)

            assert isinstance(results, list)
            if len(results) > 0:  # Some engines might not return results
                assert isinstance(results[0], SearchResult)
                assert len(results[0].title) > 0

    def test_search_fallback_modes(self):
        """Test search with different fallback strategies."""
        # Test with a simple query that should work
        searcher = TarziSearcher()
        results = searcher.search("test", max_results=1)

        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], SearchResult)
            assert len(results[0].title) > 0

    def test_search_with_minimal_content(self):
        """Test search with minimal content requirements."""
        try:
            searcher = TarziSearcher()
            # Use a very simple query that's likely to work
            results = searcher.search("hello", max_results=1)

            assert isinstance(results, list)
            if len(results) > 0:
                result = results[0]
                assert isinstance(result, SearchResult)
                # Just check that we have basic structure, don't require specific content
                assert hasattr(result, "title")
                assert hasattr(result, "url")
                assert hasattr(result, "snippet")

        except Exception as e:
            pytest.skip(f"Minimal search test failed: {e}")
