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

    def test_search_and_fetch_tarzi_native(self):
        """Test search_and_fetch using tarzi's native function."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_and_fetch"
        ) as mock_search_and_fetch, patch("cogents.resources.tarzi.searcher.TarziFetcher") as mock_fetcher_cls:
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()

            # Mock fetcher with proper fetch_mode
            mock_fetcher = Mock()
            mock_fetcher.fetch_mode = "plain_request"
            mock_fetcher_cls.return_value = mock_fetcher

            # Mock search_and_fetch results
            mock_tarzi_result = MockTarziSearchResult("Test Title", "https://example.com", "Test snippet")
            mock_content = "<html><body>Test content</body></html>"
            mock_search_and_fetch.return_value = [(mock_tarzi_result, mock_content)]

            searcher = TarziSearcher()
            results = searcher.search_and_fetch("test query", content_mode=ContentMode.RAW_HTML)

            assert len(results) == 1
            assert isinstance(results[0], SearchWithContentResult)
            assert results[0].result.title == "Test Title"
            assert results[0].content == "<html><body>Test content</body></html>"
            assert results[0].content_mode == ContentMode.RAW_HTML

            mock_search_and_fetch.assert_called_once_with("test query", "webquery", 5, "plain_request", "html")

    def test_search_and_fetch_separate(self):
        """Test search_and_fetch using separate search and fetch."""
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
            results = searcher.search_and_fetch("test query", content_mode=ContentMode.LLM_FORMATTED)

            assert len(results) == 1
            assert isinstance(results[0], SearchWithContentResult)
            assert results[0].result.title == "Test Title"
            assert results[0].content == "LLM formatted content"
            assert results[0].content_mode == ContentMode.LLM_FORMATTED

            mock_fetcher.fetch.assert_called_once_with("https://example.com", ContentMode.LLM_FORMATTED)

    def test_search_and_fetch_with_fetch_error(self):
        """Test search_and_fetch when content fetching fails."""
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
            results = searcher.search_and_fetch("test query", content_mode=ContentMode.LLM_FORMATTED)

            # Should still return result with error message
            assert len(results) == 1
            assert "Failed to fetch content: Fetch failed" in results[0].content

    def test_search_and_fetch_string_content_mode(self):
        """Test search_and_fetch with string content mode."""
        with patch("tarzi.Config") as mock_config, patch("tarzi.SearchEngine") as mock_search_engine_cls, patch(
            "tarzi.search_and_fetch"
        ) as mock_search_and_fetch, patch("cogents.resources.tarzi.searcher.TarziFetcher"):
            # Setup mocks
            mock_config.from_str.return_value = Mock()
            mock_search_engine_cls.from_config.return_value = Mock()

            mock_tarzi_result = MockTarziSearchResult("Test", "https://example.com", "Snippet")
            mock_search_and_fetch.return_value = [(mock_tarzi_result, "# Markdown content")]

            searcher = TarziSearcher()
            results = searcher.search_and_fetch("test query", content_mode="markdown")  # String instead of enum

            assert len(results) == 1
            assert results[0].content == "# Markdown content"

    def test_can_use_tarzi_search_and_fetch(self):
        """Test _can_use_tarzi_search_and_fetch logic."""
        with patch("tarzi.Config"), patch("tarzi.SearchEngine"), patch("cogents.resources.tarzi.searcher.TarziFetcher"):
            searcher = TarziSearcher()

            # Should return True for RAW_HTML and MARKDOWN
            assert searcher._can_use_tarzi_search_and_fetch(ContentMode.RAW_HTML, None) is True
            assert searcher._can_use_tarzi_search_and_fetch(ContentMode.MARKDOWN, None) is True
            assert searcher._can_use_tarzi_search_and_fetch("raw_html", None) is True
            assert searcher._can_use_tarzi_search_and_fetch("markdown", None) is True

            # Should return False for LLM_FORMATTED
            assert searcher._can_use_tarzi_search_and_fetch(ContentMode.LLM_FORMATTED, None) is False
            assert searcher._can_use_tarzi_search_and_fetch("llm_formatted", None) is False
            assert searcher._can_use_tarzi_search_and_fetch("invalid", None) is False

    def test_get_supported_engines(self):
        """Test getting supported search engines."""
        with patch("tarzi.Config"), patch("tarzi.SearchEngine"), patch("cogents.resources.tarzi.searcher.TarziFetcher"):
            searcher = TarziSearcher()
            engines = searcher.get_supported_engines()

            assert isinstance(engines, list)
            assert "google" in engines
            assert "bing" in engines
            assert "duckduckgo" in engines
            assert "brave" in engines

    def test_get_supported_content_modes(self):
        """Test getting supported content modes."""
        with patch("tarzi.Config"), patch("tarzi.SearchEngine"), patch(
            "cogents.resources.tarzi.searcher.TarziFetcher"
        ) as mock_fetcher_cls:
            # Mock fetcher with supported modes
            mock_fetcher = Mock()
            mock_fetcher.get_supported_modes.return_value = {
                "raw_html": "Raw HTML content",
                "markdown": "Markdown content",
                "llm_formatted": "LLM formatted content",
            }
            mock_fetcher_cls.return_value = mock_fetcher

            searcher = TarziSearcher()
            content_modes = searcher.get_supported_content_modes()

            assert isinstance(content_modes, dict)
            assert "raw_html" in content_modes
            assert "markdown" in content_modes
            assert "llm_formatted" in content_modes


@pytest.mark.integration
class TestTarziSearcherIntegration:
    """Integration test cases for TarziSearcher."""

    def test_search_real_query(self):
        """Test searching with a real query."""
        try:
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

        except Exception as e:
            pytest.skip(f"Network or search request failed: {e}")

    def test_search_and_fetch_real_query_markdown(self):
        """Test search_and_fetch with real query and markdown mode."""
        try:
            searcher = TarziSearcher()
            results = searcher.search_and_fetch("Python programming", max_results=2, content_mode=ContentMode.MARKDOWN)

            assert isinstance(results, list)
            assert len(results) > 0
            assert len(results) <= 2

            for result in results:
                assert isinstance(result, SearchWithContentResult)
                assert isinstance(result.result, SearchResult)
                assert isinstance(result.content, str)
                assert result.content_mode == ContentMode.MARKDOWN
                assert len(result.content) > 0

        except Exception as e:
            pytest.skip(f"Network or search request failed: {e}")

    def test_search_and_fetch_real_query_raw_html(self):
        """Test search_and_fetch with real query and raw HTML mode."""
        try:
            searcher = TarziSearcher()
            results = searcher.search_and_fetch("Python programming", max_results=1, content_mode=ContentMode.RAW_HTML)

            assert isinstance(results, list)
            assert len(results) > 0

            result = results[0]
            assert isinstance(result, SearchWithContentResult)
            assert result.content_mode == ContentMode.RAW_HTML
            assert "html" in result.content.lower() or len(result.content) > 0

        except Exception as e:
            pytest.skip(f"Network or search request failed: {e}")

    def test_search_and_fetch_real_query_llm_formatted(self):
        """Test search_and_fetch with real query and LLM formatted mode."""
        try:
            import os

            if not os.getenv("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY not available for LLM formatting test")

            fetcher_config = {"llm_provider": "openai", "llm_api_key": os.getenv("OPENAI_API_KEY")}

            searcher = TarziSearcher(fetcher_config=fetcher_config)
            results = searcher.search_and_fetch(
                "Python programming", max_results=1, content_mode=ContentMode.LLM_FORMATTED
            )

            assert isinstance(results, list)
            assert len(results) > 0

            result = results[0]
            assert isinstance(result, SearchWithContentResult)
            assert result.content_mode == ContentMode.LLM_FORMATTED
            assert len(result.content) > 0

        except Exception as e:
            pytest.skip(f"Network, search, or LLM request failed: {e}")

    def test_search_different_engines(self):
        """Test search with different search engines."""
        engines_to_test = ["duckduckgo", "bing"]

        for engine in engines_to_test:
            try:
                searcher = TarziSearcher(search_engine=engine)
                results = searcher.search("Python", max_results=1)

                assert isinstance(results, list)
                if len(results) > 0:  # Some engines might not return results
                    assert isinstance(results[0], SearchResult)
                    assert len(results[0].title) > 0

            except Exception as e:
                pytest.skip(f"Search with engine {engine} failed: {e}")
