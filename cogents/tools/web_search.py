"""
Agentic tools for web search functionality.

This module provides tools for performing web searches using Tavily Search API
and Google AI Search, designed to be used as agentic tools in the cogent-nano framework.
"""

from typing import Optional

from langchain_core.tools import tool

from cogents.common.logging import get_logger
from cogents.common.websearch.types import SearchResult

logger = get_logger(__name__)


@tool
def tavily_search(
    query: str,
    max_results: Optional[int] = 10,
    search_depth: Optional[str] = "advanced",
    include_answer: Optional[bool] = False,
    include_raw_content: Optional[bool] = False,
) -> SearchResult:
    """
    Search the web using Tavily Search API to find relevant information and sources.

    This tool performs comprehensive web searches and can optionally generate AI-powered
    summaries of the search results. It's ideal for research tasks, fact-checking,
    and gathering current information from across the internet.

    Args:
        query: The search query to execute. Be specific and descriptive for better results.
               Examples: "latest news on AI developments", "best restaurants in Paris 2024"
        max_results: Maximum number of search results to return (1-50).
                    Higher numbers provide more comprehensive coverage but may be slower.
        search_depth: Search depth level - "basic" for quick results or "advanced" for
                     more thorough, comprehensive search with better source quality.
        include_answer: When True, generates an AI-powered summary of the search results.
                       Useful for getting a quick overview of the findings.
        include_raw_content: When True, includes the full content of web pages in results.
                            Increases response size but provides more detailed information.

    Returns:
        SearchResult object containing:
        - sources: List of web sources with titles, URLs, and content snippets
        - answer: AI-generated summary (if include_answer=True)
        - query: The original search query for reference

    Example usage:
        - Research current events: Use "advanced" depth with include_answer=True
        - Quick fact-checking: Use "basic" depth with max_results=5
        - Comprehensive research: Use "advanced" depth with max_results=20 and include_raw_content=True
    """
    logger.info(f"Performing Tavily search for query: {query}")

    try:
        from cogents.common.websearch import TavilySearchWrapper

        # Initialize Tavily Search client with configuration
        config_kwargs = {
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        tavily_search = TavilySearchWrapper(**config_kwargs)

        # Perform search
        result = tavily_search.search(query=query)

        # Generate summary if requested
        # TODO: add a more sophisticated summary generation
        summary = None
        if include_answer and result.answer:
            summary = result.answer
        elif len(result.sources) > 0:
            # Create a basic summary from the top results
            top_results = result.sources[:3]
            summary = f"Found {len(result.sources)} results for '{query}'. Top results include: " + ", ".join(
                [f"'{s.title}'" for s in top_results]
            )

        return SearchResult(query=query, sources=result.sources, answer=summary)

    except Exception as e:
        logger.error(f"Error in Tavily search: {e}")
        raise RuntimeError(f"Tavily search failed: {str(e)}")


@tool
def google_ai_search(
    query: str,
    model: Optional[str] = "gemini-2.5-flash",
    temperature: Optional[float] = 0.0,
) -> SearchResult:
    """
    Search the web using Google AI Search powered by Gemini models.

    This tool leverages Google's advanced AI search capabilities to find information,
    generate comprehensive research summaries, and provide detailed citations. It's
    particularly effective for academic research, detailed analysis, and tasks requiring
    well-sourced information with proper attribution.

    Args:
        query: The search query to execute. Be specific and detailed for best results.
               Examples: "climate change impact on agriculture 2024", "machine learning trends in healthcare"
        model: The Gemini model to use for search and content generation.
               - "gemini-2.5-flash": Fast, efficient model (recommended)
               - "gemini-2.0-flash-exp": Experimental version with latest features
        temperature: Controls creativity vs accuracy in the generated content (0.0-1.0).
                    - 0.0: Most factual and consistent (recommended for research)
                    - Higher values: More creative but potentially less accurate

    Returns:
        SearchResult object containing:
        - sources: List of cited sources with URLs and metadata
        - answer: Comprehensive research summary with inline citations
        - query: The original search query for reference

    Key features:
        - Automatic citation generation with source links
        - Comprehensive research summaries
        - High-quality source selection
        - Factual accuracy with proper attribution

    Example usage:
        - Academic research: Use temperature=0.0 for maximum accuracy
        - Creative exploration: Use temperature=0.3-0.7 for more varied perspectives
        - Fact-checking: Use temperature=0.0 with specific, detailed queries
    """
    logger.info(f"Performing Google AI search for query: {query}")

    try:
        from cogents.common.websearch import GoogleAISearch

        # Initialize Google AI Search client
        google = GoogleAISearch()

        # Perform search
        return google.search(query=query, model=model, temperature=temperature)

    except Exception as e:
        logger.error(f"Error in Google AI search: {e}")
        raise RuntimeError(f"Google AI search failed: {str(e)}")
