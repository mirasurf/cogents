"""
Example demonstrating the use of web search tools.

This example shows how to use the Tavily search and Google AI search tools
from the search toolkit.
"""

import asyncio

from cogents.tools.toolkits.search_toolkit import SearchToolkit


async def example_tavily_search():
    """Example of using Tavily search tool."""
    print("=== Tavily Search Example ===")

    try:
        # Initialize search toolkit
        search_toolkit = SearchToolkit()

        # Perform a basic search
        result = await search_toolkit.tavily_search(
            query="best travel destinations 2024",
            max_results=5,
            search_depth="advanced",
            include_answer=True,
        )

        print(f"Query: {result.query}")
        print(f"Number of Results: {len(result.sources)}")

        if result.answer:
            print(f"\nSummary: {result.answer}")

        print("\nTop Results:")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"{i}. {source.title}")
            print(f"   URL: {source.url}")
            if hasattr(source, "content"):
                print(f"   Content: {source.content[:100]}...")
            print()

    except Exception as e:
        print(f"Error in Tavily search: {e}")


async def example_google_ai_search():
    """Example of using Google AI search tool."""
    print("=== Google AI Search Example ===")

    try:
        # Initialize search toolkit
        search_toolkit = SearchToolkit()

        # Perform a search with Google AI
        result = await search_toolkit.google_ai_search(
            query="latest travel trends 2024",
            model="gemini-2.0-flash-exp",
            temperature=0.0,
        )

        print(f"Query: {result.query}")
        print(f"Number of Sources: {len(result.sources)}")

        if result.answer:
            print(f"\nResearch Summary:")
            print(result.answer)

        print("\nSources:")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"{i}. {source.title}")
            if hasattr(source, "label"):
                print(f"   Label: {source.label}")
            print(f"   URL: {source.url}")
            print()

    except Exception as e:
        print(f"Error in Google AI search: {e}")


async def example_unified_web_search():
    """Example of using both search tools."""
    print("=== Combined Search Example ===")

    try:
        # Initialize search toolkit
        search_toolkit = SearchToolkit()

        # Search with Tavily
        print("Searching with Tavily...")
        tavily_result = await search_toolkit.tavily_search(
            query="budget travel tips", max_results=3, include_answer=True
        )

        print(f"Tavily Results: {len(tavily_result.sources)} results")
        if tavily_result.answer:
            print(f"Summary: {tavily_result.answer[:200]}...")

        print("\n" + "=" * 50 + "\n")

        # Search with Google AI
        print("Searching with Google AI...")
        google_result = await search_toolkit.google_ai_search(query="budget travel tips", temperature=0.0)

        print(f"Google AI Results: {len(google_result.sources)} results")
        if google_result.answer:
            print(f"Summary: {google_result.answer[:200]}...")

    except Exception as e:
        print(f"Error in combined search: {e}")


async def main():
    """Run all examples."""
    print("Web Search Tools Examples")
    print("=" * 50)

    # Run examples
    await example_tavily_search()
    print("\n" + "=" * 50 + "\n")

    await example_google_ai_search()
    print("\n" + "=" * 50 + "\n")

    await example_unified_web_search()


if __name__ == "__main__":
    asyncio.run(main())
