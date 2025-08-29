"""
Example demonstrating the use of web search tools.

This example shows how to use the Tavily search and Google AI search tools
from the cogents.tools.web_search module.
"""

import asyncio

from cogents.toolify.repo.web_search import google_ai_search, tavily_search


def example_tavily_search():
    """Example of using Tavily search tool."""
    print("=== Tavily Search Example ===")

    try:
        # Perform a basic search
        result = tavily_search.invoke(
            {
                "query": "best travel destinations 2024",
                "max_results": 5,
                "search_depth": "advanced",
                "include_answer": True,
            }
        )

        print(f"Query: {result['query']}")
        print(f"Search Type: {result['search_type']}")
        print(f"Number of Results: {len(result['results'])}")

        if result["summary"]:
            print(f"\nSummary: {result['summary']}")

        print("\nTop Results:")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"{i}. {source['title']}")
            print(f"   URL: {source['url']}")
            print(f"   Content: {source['content'][:100]}...")
            print()

    except Exception as e:
        print(f"Error in Tavily search: {e}")


def example_google_ai_search():
    """Example of using Google AI search tool."""
    print("=== Google AI Search Example ===")

    try:
        # Perform a search with Google AI
        result = google_ai_search.invoke(
            {
                "query": "latest travel trends 2024",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.0,
            }
        )

        print(f"Query: {result['query']}")
        print(f"Search Type: {result['search_type']}")
        print(f"Number of Sources: {len(result['sources'])}")

        if result["summary"]:
            print(f"\nResearch Summary:")
            print(result["summary"])

        print("\nSources:")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"{i}. {source['title']}")
            print(f"   Label: {source['label']}")
            print(f"   URL: {source['url']}")
            print()

    except Exception as e:
        print(f"Error in Google AI search: {e}")


def example_unified_web_search():
    """Example of using both search tools."""
    print("=== Combined Search Example ===")

    try:
        # Search with Tavily
        print("Searching with Tavily...")
        tavily_result = tavily_search.invoke({"query": "budget travel tips", "max_results": 3, "include_answer": True})

        print(f"Tavily Results: {len(tavily_result['results'])} results")
        if tavily_result["summary"]:
            print(f"Summary: {tavily_result['summary'][:200]}...")

        print("\n" + "=" * 50 + "\n")

        # Search with Google AI
        print("Searching with Google AI...")
        google_result = google_ai_search.invoke({"query": "budget travel tips", "temperature": 0.0})

        print(f"Google AI Results: {len(google_result['results'])} results")
        if google_result["summary"]:
            print(f"Summary: {google_result['summary'][:200]}...")

    except Exception as e:
        print(f"Error in combined search: {e}")


async def main():
    """Run all examples."""
    print("Web Search Tools Examples")
    print("=" * 50)

    # Run examples
    example_tavily_search()
    print("\n" + "=" * 50 + "\n")

    example_google_ai_search()
    print("\n" + "=" * 50 + "\n")

    example_unified_web_search()


if __name__ == "__main__":
    asyncio.run(main())
