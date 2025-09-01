#!/usr/bin/env python3
import os
import sys

# Add the project root to the path so we can import cogents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cogents.resources.tarzi.fetcher import ContentMode, TarziFetcher


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
        try:
            llm_fetcher = TarziFetcher(llm_provider="openrouter")
            llm_content = llm_fetcher.fetch(test_url, ContentMode.LLM_FORMATTED)
            print(f"   Success! Content length: {len(llm_content)} characters")
            print(f"   Preview: {llm_content[:100]}...")
        except Exception as e:
            print(f"   Failed: {e}")

    except Exception as e:
        print(f"Content fetching demo failed: {e}")


def main():
    """Run all demos."""
    try:
        # Run all demos
        demo_content_fetching()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
