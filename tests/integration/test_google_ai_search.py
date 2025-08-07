"""
Integration tests for Google AI Search functionality.
"""

import os

import pytest

from cogents.common.websearch.google_ai_search import GoogleAISearch
from cogents.common.websearch.types import SearchResult, SourceItem


@pytest.mark.integration
@pytest.mark.slow
class TestGoogleAISearchIntegration:
    """Integration tests that require actual API access."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set - skipping integration test",
    )
    def test_real_search_integration(self):
        """Test real search with actual API (requires GEMINI_API_KEY)."""
        google_search = GoogleAISearch()

        result = google_search.search(
            query="latest innovation news about WAIC 2025 in shanghai",
            model="gemini-2.5-flash",
            temperature=0,
            research_id=1,
        )

        # Verify the result structure
        assert isinstance(result, SearchResult)
        assert len(result.query) == 1
        assert "WAIC 2025" in result.query[0]
        assert len(result.answer) == 1
        assert len(result.answer[0]) > 0

        # Verify sources if available
        if result.sources:
            for source in result.sources:
                assert isinstance(source, SourceItem)
                assert source.label is not None
