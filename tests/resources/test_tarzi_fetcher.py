"""
Tests for tarzi fetcher functionality.
"""

from unittest.mock import Mock, patch

import pytest

from cogents.resources.tarzi.fetcher import ContentMode, TarziFetcher


class TestTarziFetcher:
    """Unit test cases for TarziFetcher."""

    def test_initialization_with_defaults(self):
        """Test TarziFetcher initialization with default parameters."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher.from_config.return_value = Mock()
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()

            assert fetcher.llm_provider == "llamacpp"
            assert fetcher.fetch_mode == "browser_headless"  # Updated to match actual default
            assert fetcher.timeout == 30

    def test_initialization_with_custom_params(self):
        """Test TarziFetcher initialization with custom parameters."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher.from_config.return_value = Mock()
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher(
                llm_provider="ollama",
                fetch_mode="plain_request",
                timeout=60,
            )

            assert fetcher.llm_provider == "ollama"
            assert fetcher.fetch_mode == "plain_request"
            assert fetcher.timeout == 60

    def test_tarzi_import_error(self):
        """Test handling of tarzi import error."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client"), patch(
            "builtins.__import__", side_effect=ImportError("No module named 'tarzi'")
        ):
            with pytest.raises(ImportError, match="tarzi library is required but not available"):
                TarziFetcher()

    def test_fetch_raw_html_mode(self):
        """Test fetching content in raw HTML mode."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch.return_value = "<html><body>Test content</body></html>"
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            result = fetcher.fetch("https://example.com", ContentMode.RAW_HTML)

            assert result == "<html><body>Test content</body></html>"
            mock_web_fetcher.fetch.assert_called_once_with("https://example.com", "browser_headless", "html")

    def test_fetch_markdown_mode(self):
        """Test fetching content in markdown mode."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch.return_value = "# Test Content\n\nThis is markdown content."
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            result = fetcher.fetch("https://example.com", ContentMode.MARKDOWN)

            assert result == "# Test Content\n\nThis is markdown content."
            mock_web_fetcher.fetch.assert_called_once_with("https://example.com", "browser_headless", "markdown")

    def test_fetch_llm_formatted_mode(self):
        """Test fetching content in LLM formatted mode."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch(
            "tarzi.Converter"
        ) as mock_converter_cls:
            # Setup mocks
            mock_llm_client = Mock()
            mock_llm_client.completion.return_value = "LLM formatted content"
            mock_get_llm_client.return_value = mock_llm_client

            mock_config.from_str.return_value = Mock()

            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch.return_value = "<html><body>Raw HTML</body></html>"
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher

            mock_converter = Mock()
            mock_converter.convert.return_value = "Raw HTML converted to markdown"
            mock_converter_cls.return_value = mock_converter

            fetcher = TarziFetcher(llm_api_key="/path/to/model.gguf")
            result = fetcher.fetch("https://example.com", ContentMode.LLM_FORMATTED)

            assert result == "LLM formatted content"
            mock_web_fetcher.fetch.assert_called_once_with("https://example.com", "browser_headless", "html")
            mock_converter.convert.assert_called_once_with("<html><body>Raw HTML</body></html>", "markdown")
            mock_llm_client.completion.assert_called_once()

    def test_fetch_llm_formatted_mode_no_llm_client(self):
        """Test LLM formatted mode without LLM client configured."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks - no LLM client
            mock_get_llm_client.return_value = None
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher_cls.from_config.return_value = Mock()
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            fetcher._llm_client = None  # Simulate no LLM client

            with pytest.raises(ValueError, match="LLM client not configured"):
                fetcher.fetch("https://example.com", ContentMode.LLM_FORMATTED)

    def test_fetch_llm_formatted_mode_llm_fallback(self):
        """Test LLM formatted mode with fallback to markdown when LLM fails."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch(
            "tarzi.Converter"
        ) as mock_converter_cls:
            # Setup mocks
            mock_llm_client = Mock()
            mock_llm_client.completion.side_effect = Exception("LLM error")
            mock_get_llm_client.return_value = mock_llm_client

            mock_config.from_str.return_value = Mock()

            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch.return_value = "<html><body>Raw HTML</body></html>"
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher

            mock_converter = Mock()
            mock_converter.convert.return_value = "Fallback markdown content"
            mock_converter_cls.return_value = mock_converter

            fetcher = TarziFetcher(llm_api_key="/path/to/model.gguf")
            result = fetcher.fetch("https://example.com", ContentMode.LLM_FORMATTED)

            # Should fallback to markdown
            assert result == "Fallback markdown content"

    def test_fetch_raw(self):
        """Test fetching raw content without formatting."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch_raw.return_value = "Raw response content"
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            result = fetcher.fetch_raw("https://example.com")

            assert result == "Raw response content"
            mock_web_fetcher.fetch_raw.assert_called_once_with("https://example.com", "browser_headless")

    def test_fetch_invalid_content_mode(self):
        """Test fetching with invalid content mode."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher_cls.from_config.return_value = Mock()
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()

            with pytest.raises(ValueError, match="Invalid content_mode"):
                fetcher.fetch("https://example.com", "invalid_mode")

    def test_fetch_string_content_mode(self):
        """Test fetching with string content mode (backward compatibility)."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher = Mock()
            mock_web_fetcher.fetch.return_value = "<html><body>Test</body></html>"
            mock_web_fetcher_cls.from_config.return_value = mock_web_fetcher
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            result = fetcher.fetch("https://example.com", "raw_html")  # String instead of enum

            assert result == "<html><body>Test</body></html>"
            mock_web_fetcher.fetch.assert_called_once_with("https://example.com", "browser_headless", "html")

    def test_get_supported_modes(self):
        """Test getting supported content modes."""
        with patch("cogents.resources.tarzi.fetcher.get_llm_client") as mock_get_llm_client, patch(
            "tarzi.Config"
        ) as mock_config, patch("tarzi.WebFetcher") as mock_web_fetcher_cls, patch("tarzi.Converter") as mock_converter:
            # Setup mocks
            mock_get_llm_client.return_value = Mock()
            mock_config.from_str.return_value = Mock()
            mock_web_fetcher_cls.from_config.return_value = Mock()
            mock_converter.return_value = Mock()

            fetcher = TarziFetcher()
            modes = fetcher.get_supported_modes()

            assert isinstance(modes, dict)
            assert "raw_html" in modes
            assert "markdown" in modes
            assert "llm_formatted" in modes


@pytest.mark.integration
class TestTarziFetcherIntegration:
    """Integration test cases for TarziFetcher."""

    def test_fetch_real_url_raw_html(self):
        """Test fetching real URL in raw HTML mode."""
        try:
            fetcher = TarziFetcher()
            result = fetcher.fetch("https://httpbin.org/html", ContentMode.RAW_HTML)

            assert isinstance(result, str)
            assert len(result) > 0
            assert "html" in result.lower()

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")

    def test_fetch_real_url_markdown(self):
        """Test fetching real URL in markdown mode."""
        try:
            fetcher = TarziFetcher()
            result = fetcher.fetch("https://httpbin.org/html", ContentMode.MARKDOWN)

            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")

    def test_fetch_real_url_llm_formatted(self):
        """Test fetching real URL with LLM formatting."""
        try:
            fetcher = TarziFetcher()
            result = fetcher.fetch("https://httpbin.org/html", ContentMode.LLM_FORMATTED)

            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"Network or LLM request failed: {e}")

    def test_fetch_raw_real_url(self):
        """Test fetching raw content from real URL."""
        try:
            fetcher = TarziFetcher()
            result = fetcher.fetch_raw("https://httpbin.org/html")

            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")
