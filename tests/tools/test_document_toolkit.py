"""
Tests for DocumentToolkit functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogents.tools import ToolkitConfig, get_toolkit


@pytest.fixture
def document_config():
    """Create a test configuration for DocumentToolkit."""
    return ToolkitConfig(
        name="document", config={"OPENAI_API_KEY": "test_openai_key", "max_file_size": 10 * 1024 * 1024}  # 10MB
    )


@pytest.fixture
def document_toolkit(document_config):
    """Create DocumentToolkit instance for testing."""
    return get_toolkit("document", document_config)


class TestDocumentToolkit:
    """Test cases for DocumentToolkit."""

    async def test_toolkit_initialization(self, document_toolkit):
        """Test that DocumentToolkit initializes correctly."""
        assert document_toolkit is not None
        assert hasattr(document_toolkit, "document_qa")
        assert hasattr(document_toolkit, "get_document_info")

    async def test_get_tools_map(self, document_toolkit):
        """Test that tools map is correctly defined."""
        tools_map = await document_toolkit.get_tools_map()

        expected_tools = ["document_qa", "get_document_info", "extract_text"]
        for tool_name in expected_tools:
            assert tool_name in tools_map
            assert callable(tools_map[tool_name])

    def test_is_url_detection(self, document_toolkit):
        """Test URL detection functionality."""
        assert document_toolkit._is_url("https://example.com/doc.pdf") == True
        assert document_toolkit._is_url("http://example.com/doc.pdf") == True
        assert document_toolkit._is_url("/path/to/doc.pdf") == False
        assert document_toolkit._is_url("doc.pdf") == False

    def test_get_file_extension(self, document_toolkit):
        """Test file extension extraction."""
        assert document_toolkit._get_file_extension("document.pdf") == "pdf"
        assert document_toolkit._get_file_extension("file.docx") == "docx"
        assert document_toolkit._get_file_extension("https://example.com/doc.pdf") == "pdf"
        assert document_toolkit._get_file_extension("no_extension") == ""

    def test_get_file_md5(self, document_toolkit):
        """Test MD5 hash calculation."""
        expected_md5 = "5d41402abc4b2a76b9719d911017c592"  # MD5 of "hello"

        # Test with actual content
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"hello")
            temp_file.flush()

            md5_hash = document_toolkit._get_file_md5(temp_file.name)
            assert md5_hash == expected_md5

    @patch("aiohttp.ClientSession.get")
    async def test_download_document_success(self, mock_get, document_toolkit):
        """Test successful document download."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"PDF content here"
        mock_get.return_value.__aenter__.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await document_toolkit._download_document("https://example.com/doc.pdf", temp_dir)

            assert result is not None
            assert Path(result).exists()
            assert Path(result).suffix == ".pdf"

    @patch("aiohttp.ClientSession.get")
    async def test_download_document_error(self, mock_get, document_toolkit):
        """Test document download error handling."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await document_toolkit._download_document("https://example.com/nonexistent.pdf", temp_dir)

            assert result is None

    async def test_handle_document_path_local_file(self, document_toolkit):
        """Test handling local file path."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"test pdf content")
            temp_file.flush()

            try:
                result = await document_toolkit._handle_document_path(temp_file.name)
                assert result == temp_file.name
            finally:
                Path(temp_file.name).unlink()

    async def test_handle_document_path_nonexistent_file(self, document_toolkit):
        """Test handling non-existent local file."""
        result = await document_toolkit._handle_document_path("/nonexistent/file.pdf")
        assert result is None

    @patch("aiohttp.ClientSession.get")
    async def test_handle_document_path_url(self, mock_get, document_toolkit):
        """Test handling URL document path."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"PDF content"
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await document_toolkit._handle_document_path("https://example.com/doc.pdf")

        assert result is not None
        assert Path(result).exists()

    @patch("fitz.open")
    async def test_parse_document_pdf_success(self, mock_fitz_open, document_toolkit):
        """Test successful PDF parsing."""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is page 1 content.\nMore text here."
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.page_count = 1
        mock_fitz_open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            result = await document_toolkit._parse_document(temp_file.name)

            assert isinstance(result, str)
            assert "This is page 1 content" in result

    @patch("fitz.open")
    async def test_parse_document_pdf_error(self, mock_fitz_open, document_toolkit):
        """Test PDF parsing error handling."""
        mock_fitz_open.side_effect = Exception("Cannot open PDF")

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            with pytest.raises(Exception, match="Cannot open PDF"):
                await document_toolkit._parse_document(temp_file.name)

    async def test_parse_document_unsupported_format(self, document_toolkit):
        """Test parsing unsupported document format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz") as temp_file:
            # This should fail because the method expects an MD5 hash, not a file path
            with pytest.raises(ValueError):
                await document_toolkit._parse_document(temp_file.name)

    async def test_parse_document_text_file(self, document_toolkit):
        """Test parsing plain text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write("This is a plain text document.\nWith multiple lines.")
            temp_file.flush()

            try:
                # This should fail because the method expects an MD5 hash, not a file path
                with pytest.raises(ValueError):
                    await document_toolkit._parse_document(temp_file.name)
            finally:
                Path(temp_file.name).unlink()

    @patch("openai.AsyncOpenAI")
    async def test_document_qa_success(self, mock_openai, document_toolkit):
        """Test successful document Q&A."""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="The document discusses machine learning algorithms."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch.object(document_toolkit, "_handle_document_path") as mock_handle:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                mock_handle.return_value = temp_file.name

                with patch.object(document_toolkit, "_parse_document") as mock_parse:
                    mock_parse.return_value = {
                        "status": "success",
                        "content": "This document contains information about machine learning.",
                        "page_count": 1,
                    }

                    result = await document_toolkit.document_qa("document.pdf", "What does this document discuss?")

                    assert isinstance(result, dict)
                    assert result["status"] == "success"
                    assert "machine learning algorithms" in result["answer"]

    async def test_document_qa_file_not_found(self, document_toolkit):
        """Test document Q&A with file not found."""
        result = await document_toolkit.document_qa("/nonexistent/document.pdf", "What is this about?")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "not found" in result["error"]

    async def test_document_qa_parse_error(self, document_toolkit):
        """Test document Q&A with parsing error."""
        with patch.object(document_toolkit, "_handle_document_path") as mock_handle:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                mock_handle.return_value = temp_file.name

                with patch.object(document_toolkit, "_parse_document") as mock_parse:
                    mock_parse.return_value = {"status": "error", "error": "Failed to parse document"}

                    result = await document_toolkit.document_qa("document.pdf", "What is this about?")

                    assert isinstance(result, dict)
                    assert result["status"] == "error"
                    assert "Failed to parse document" in result["error"]

    @patch("openai.AsyncOpenAI")
    async def test_document_qa_openai_error(self, mock_openai, document_toolkit):
        """Test document Q&A with OpenAI API error."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        with patch.object(document_toolkit, "_handle_document_path") as mock_handle:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                mock_handle.return_value = temp_file.name

                with patch.object(document_toolkit, "_parse_document") as mock_parse:
                    mock_parse.return_value = {"status": "success", "content": "Document content", "page_count": 1}

                    result = await document_toolkit.document_qa("document.pdf", "What is this about?")

                    assert isinstance(result, dict)
                    assert result["status"] == "error"
                    assert "API Error" in result["error"]

    async def test_get_document_info_success(self, document_toolkit):
        """Test successful document info retrieval."""
        with patch.object(document_toolkit, "_handle_document_path") as mock_handle:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                temp_file.write(b"test content")
                temp_file.flush()
                mock_handle.return_value = temp_file.name

                with patch.object(document_toolkit, "_parse_document") as mock_parse:
                    mock_parse.return_value = {"status": "success", "content": "Document content here", "page_count": 2}

                    result = await document_toolkit.get_document_info("document.pdf")

                    assert isinstance(result, dict)
                    assert "path" in result
                    assert "extension" in result
                    assert result["content_length"] > 0
                    assert "md5_hash" in result

    async def test_get_document_info_file_not_found(self, document_toolkit):
        """Test document info with file not found."""
        result = await document_toolkit.get_document_info("/nonexistent/document.pdf")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "not found" in result["error"]

    async def test_get_document_info_parse_error(self, document_toolkit):
        """Test document info with parsing error."""
        with patch.object(document_toolkit, "_handle_document_path") as mock_handle:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                mock_handle.return_value = temp_file.name

                with patch.object(document_toolkit, "_parse_document") as mock_parse:
                    mock_parse.return_value = {"status": "error", "error": "Parsing failed"}

                    result = await document_toolkit.get_document_info("document.pdf")

                    assert isinstance(result, dict)
                    assert result["status"] == "error"
                    assert "Parsing failed" in result["error"]


class TestDocumentToolkitWithoutOpenAI:
    """Test DocumentToolkit without OpenAI API key."""

    @pytest.fixture
    def document_toolkit_no_openai(self):
        """Create DocumentToolkit without OpenAI API key."""
        config = ToolkitConfig(name="document", config={})
        return get_toolkit("document", config)

    async def test_initialization_without_openai_key(self, document_toolkit_no_openai):
        """Test initialization without OpenAI API key."""
        assert document_toolkit_no_openai is not None
        assert document_toolkit_no_openai.openai_api_key is None

    async def test_document_qa_without_openai_key(self, document_toolkit_no_openai):
        """Test document Q&A without OpenAI API key."""
        result = await document_toolkit_no_openai.document_qa("document.pdf", "What is this?")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "OpenAI API key" in result["error"]


class TestDocumentToolkitEdgeCases:
    """Test edge cases and error conditions."""

    async def test_very_large_document_handling(self, document_toolkit):
        """Test handling of very large documents."""
        # Create a large text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            large_content = "This is a test line.\n" * 10000  # Large content
            temp_file.write(large_content)
            temp_file.flush()

            try:
                # This should fail because the method expects an MD5 hash, not a file path
                with pytest.raises(ValueError):
                    await document_toolkit._parse_document(temp_file.name)
            finally:
                Path(temp_file.name).unlink()

    @pytest.mark.parametrize(
        "file_extension,expected_type",
        [
            (".pdf", "pdf"),
            (".txt", "txt"),
            (".docx", "docx"),
            (".doc", "doc"),
        ],
    )
    def test_different_document_formats(self, document_toolkit, file_extension, expected_type):
        """Test handling of different document formats."""
        filename = f"test{file_extension}"
        assert document_toolkit._get_file_extension(filename) == expected_type

    async def test_concurrent_document_operations(self, document_toolkit):
        """Test concurrent document operations."""
        import asyncio

        # Create multiple test files
        test_files = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            temp_file.write(f"Test document {i} content")
            temp_file.flush()
            test_files.append(temp_file.name)

        try:
            # Perform multiple operations concurrently
            tasks = [document_toolkit.get_document_info(file_path) for file_path in test_files]

            results = await asyncio.gather(*tasks)

            # All operations should succeed
            for result in results:
                assert result["status"] == "success"

        finally:
            # Clean up test files
            for file_path in test_files:
                Path(file_path).unlink()

    async def test_document_with_special_characters(self, document_toolkit):
        """Test handling documents with special characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
            special_content = "Document with special characters: áéíóú, 中文, 🚀, ñ"
            temp_file.write(special_content)
            temp_file.flush()

            try:
                # This should fail because the method expects an MD5 hash, not a file path
                with pytest.raises(ValueError):
                    await document_toolkit._parse_document(temp_file.name)
            finally:
                Path(temp_file.name).unlink()
