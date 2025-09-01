#!/usr/bin/env python3
"""
Tarzi Search and Fetch Example

This example demonstrates how to use the tarzi-based web search and content fetching
capabilities in cogents. It shows:

1. Basic web search functionality
2. Content fetching in different formats (raw HTML, markdown, LLM formatted)
3. Search with content fetching
4. Different search engine configurations
5. Error handling and fallback strategies

Requirements:
- tarzi library installed
- For LLM formatting: OpenAI API key or other LLM provider configured
- For browser-based fetching: appropriate browser drivers installed
"""

import os
import sys

# Add the project root to the path so we can import cogents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cogents.resources.tarzi.fetcher import ContentMode, TarziFetcher
from cogents.resources.tarzi.searcher import TarziSearcher


def demo_basic_search():
    """Demonstrate basic web search functionality."""
    print("\n" + "=" * 60)
    print("BASIC WEB SEARCH DEMO")
    print("=" * 60)

    try:
        # Initialize searcher with default settings (DuckDuckGo)
        searcher = TarziSearcher()

        # Perform a basic search
        query = "Python programming tutorials"
        print(f"Searching for: '{query}'")

        results = searcher.search(query, max_results=3)

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print(f"   Rank: {result.rank}")

    except Exception as e:
        print(f"Search failed: {e}")
        print("This might be due to missing browser drivers or network issues.")


def demo_different_search_engines():
    """Demonstrate search with different search engines."""
    print("\n" + "=" * 60)
    print("DIFFERENT SEARCH ENGINES DEMO")
    print("=" * 60)

    search_engines = ["duckduckgo", "bing", "google"]

    for engine in search_engines:
        try:
            print(f"\nTrying search engine: {engine}")
            searcher = TarziSearcher(search_engine=engine)

            results = searcher.search("machine learning", max_results=2)
            print(f"  Found {len(results)} results")

            if results:
                print(f"  First result: {results[0].title}")

        except Exception as e:
            print(f"  Failed with {engine}: {e}")


def demo_content_fetching():
    """Demonstrate content fetching from URLs."""
    print("\n" + "=" * 60)
    print("CONTENT FETCHING DEMO")
    print("=" * 60)

    # Test URL - a simple, reliable page
    test_url = "https://httpbin.org/html"

    try:
        # Initialize fetcher with default settings
        fetcher = TarziFetcher()

        print(f"Fetching content from: {test_url}")

        # Fetch raw HTML
        print("\n1. Raw HTML mode:")
        try:
            raw_html = fetcher.fetch(test_url, ContentMode.RAW_HTML)
            print(f"   Success! Content length: {len(raw_html)} characters")
            print(f"   Preview: {raw_html[:100]}...")
        except Exception as e:
            print(f"   Failed: {e}")

        # Fetch markdown
        print("\n2. Markdown mode:")
        try:
            markdown_content = fetcher.fetch(test_url, ContentMode.MARKDOWN)
            print(f"   Success! Content length: {len(markdown_content)} characters")
            print(f"   Preview: {markdown_content[:100]}...")
        except Exception as e:
            print(f"   Failed: {e}")

        # Fetch LLM formatted (if API key available)
        print("\n3. LLM Formatted mode:")
        if os.getenv("OPENAI_API_KEY"):
            try:
                llm_fetcher = TarziFetcher(llm_provider="openai")
                llm_content = llm_fetcher.fetch(test_url, ContentMode.LLM_FORMATTED)
                print(f"   Success! Content length: {len(llm_content)} characters")
                print(f"   Preview: {llm_content[:100]}...")
            except Exception as e:
                print(f"   Failed: {e}")
        else:
            print("   Skipped: OPENAI_API_KEY not available")

    except Exception as e:
        print(f"Content fetching demo failed: {e}")


def demo_search_with_content():
    """Demonstrate search with content fetching."""
    print("\n" + "=" * 60)
    print("SEARCH WITH CONTENT DEMO")
    print("=" * 60)

    try:
        # Initialize searcher
        searcher = TarziSearcher()

        query = "Python web scraping"
        print(f"Searching for: '{query}' with content fetching")

        # Search with markdown content
        print("\n1. Search with markdown content:")
        try:
            results = searcher.search_with_content(query, max_results=2, content_mode=ContentMode.MARKDOWN)

            print(f"   Found {len(results)} results with content")
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"     Title: {result.result.title}")
                print(f"     URL: {result.result.url}")
                print(f"     Content length: {len(result.content)} characters")
                print(f"     Content preview: {result.content[:100]}...")

        except Exception as e:
            print(f"   Failed: {e}")

        # Search with raw HTML content
        print("\n2. Search with raw HTML content:")
        try:
            results = searcher.search_with_content(query, max_results=1, content_mode=ContentMode.RAW_HTML)

            if results:
                result = results[0]
                print(f"   Found result: {result.result.title}")
                print(f"   Content length: {len(result.content)} characters")
                print(f"   Content preview: {result.content[:100]}...")

        except Exception as e:
            print(f"   Failed: {e}")

    except Exception as e:
        print(f"Search with content demo failed: {e}")


def demo_custom_fetcher_config():
    """Demonstrate custom fetcher configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM FETCHER CONFIGURATION DEMO")
    print("=" * 60)

    try:
        # Custom fetcher configuration
        fetcher_config = {
            "llm_provider": "ollama",  # Use Ollama if available
            "fetch_mode": "plain_request",  # Use plain HTTP requests
            "timeout": 60,  # Increase timeout
        }

        print("Creating searcher with custom fetcher configuration:")
        for key, value in fetcher_config.items():
            print(f"  {key}: {value}")

        searcher = TarziSearcher(fetcher_config=fetcher_config)

        # Test the custom configuration
        query = "data science"
        print(f"\nTesting with query: '{query}'")

        results = searcher.search_with_content(query, max_results=1, content_mode=ContentMode.MARKDOWN)

        if results:
            print("   Success! Custom configuration working.")
            result = results[0]
            print(f"   Found: {result.result.title}")
            print(f"   Content length: {len(result.content)} characters")
        else:
            print("   No results found")

    except Exception as e:
        print(f"Custom configuration demo failed: {e}")


def demo_error_handling():
    """Demonstrate error handling and fallback strategies."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    print("Testing various error scenarios:")

    # Test with invalid URL
    print("\n1. Testing with invalid URL:")
    try:
        fetcher = TarziFetcher()
        content = fetcher.fetch("https://invalid-url-that-does-not-exist.com", ContentMode.RAW_HTML)
        print("   Unexpected success!")
    except Exception as e:
        print(f"   Expected failure: {e}")

    # Test with unsupported content mode
    print("\n2. Testing with unsupported content mode:")
    try:
        fetcher = TarziFetcher()
        content = fetcher.fetch("https://example.com", "invalid_mode")
        print("   Unexpected success!")
    except Exception as e:
        print(f"   Expected failure: {e}")

    # Test search with empty query
    print("\n3. Testing search with empty query:")
    try:
        searcher = TarziSearcher()
        results = searcher.search("", max_results=5)
        print(f"   Unexpected success: {len(results)} results")
    except Exception as e:
        print(f"   Expected failure: {e}")


def main():
    """Run all demos."""
    print("TARZI SEARCH AND FETCH EXAMPLES")
    print("=" * 60)
    print("This example demonstrates the tarzi-based web search and content")
    print("fetching capabilities in cogents.")
    print("\nNote: Some features may require:")
    print("- Browser drivers for web automation")
    print("- API keys for LLM providers")
    print("- Network connectivity for web requests")

    try:
        # Run all demos
        demo_basic_search()
        demo_different_search_engines()
        demo_content_fetching()
        demo_search_with_content()
        demo_custom_fetcher_config()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED")
        print("=" * 60)
        print("Check the output above to see which features worked")
        print("and which ones encountered issues.")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
