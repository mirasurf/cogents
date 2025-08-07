import pytest

from cogents.common.websearch import TavilySearchWrapper


@pytest.mark.integration
class TestTavilySearch:
    def test_tavily_search(self):
        # Create search wrapper
        wrapper = TavilySearchWrapper()

        # Perform a search
        query = "Python programming"
        response = wrapper.search(query)

        print(f"Search query: {query}")
        assert len(response.sources) > 0, "Should have found at least one result"

        # Show first result
        if response.sources:
            result = response.sources[0]
            print(f"\nFirst result:")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Content: {result.content[:200]}...")
