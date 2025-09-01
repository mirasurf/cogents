"""
Tarzi-based web search functionality with content fetching support.

Provides web search capabilities using the tarzi library with support for:
- Basic web search operations
- Search with content fetching using multiple formatting modes
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from cogents.common.logging import get_logger

from .fetcher import ContentMode, TarziFetcher

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str
    rank: int = 0


@dataclass
class SearchWithContentResult:
    """Represents a search result with fetched content."""

    result: SearchResult
    content: str
    content_mode: ContentMode


class TarziSearcher:
    """
    Web search functionality using tarzi library.

    Provides both basic search operations and search with content fetching,
    supporting multiple content formatting modes through TarziFetcher integration.
    """

    def __init__(
        self,
        search_engine: str = "duckduckgo",
        fetcher_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TarziSearcher.

        Args:
            search_engine: Search engine to use ("google", "bing", "duckduckgo", etc.)
            fetcher_config: Configuration for the TarziFetcher (used in search_and_fetch)
        """
        self.search_engine = search_engine
        self.search_mode = "webquery"  # Always use webquery mode

        # Initialize tarzi components
        self._setup_tarzi()

        # Initialize TarziFetcher for search_and_fetch operations
        fetcher_config = fetcher_config or {}
        self._fetcher = TarziFetcher(**fetcher_config)

    def _setup_tarzi(self) -> None:
        """Setup tarzi search components with configuration."""
        try:
            # Import tarzi (lazy import to avoid dependency issues)
            import tarzi

            # Create tarzi configuration with search engine settings
            config_str = f"""
[search]
engine = "{self.search_engine}"
autoswitch = "smart"
limit = 10
"""
            self._config = tarzi.Config.from_str(config_str)
            self._search_engine = tarzi.SearchEngine.from_config(self._config)

            logger.info(f"Initialized tarzi search engine with: {self.search_engine}")
        except ImportError as e:
            logger.error(f"Failed to import tarzi: {e}")
            raise ImportError("tarzi library is required but not available") from e
        except Exception as e:
            logger.error(f"Failed to setup tarzi search: {e}")
            raise

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """
        Perform a web search and return search results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            Exception: If search fails
        """
        logger.info(f"Searching for '{query}' with mode: {self.search_mode}, max_results: {max_results}")

        try:
            # Use tarzi search_web function for direct search
            import tarzi

            tarzi_results = tarzi.search_web(query, self.search_mode, max_results)

            # Convert tarzi results to our SearchResult format
            results = []
            for i, result in enumerate(tarzi_results):
                search_result = SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    rank=getattr(result, "rank", i + 1),  # Use rank if available, else use index
                )
                results.append(search_result)

            logger.info(f"Found {len(results)} search results")
            return results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 5,
        content_mode: ContentMode = ContentMode.LLM_FORMATTED,
        fetch_mode: Optional[str] = None,
        **fetch_kwargs: Any,
    ) -> List[SearchWithContentResult]:
        """
        Perform a web search and fetch content from the results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            content_mode: Content formatting mode for fetched content
            fetch_mode: Override default fetch mode for tarzi
            **fetch_kwargs: Additional arguments passed to content fetching

        Returns:
            List of SearchWithContentResult objects

        Raises:
            Exception: If search or fetching fails
        """
        content_mode_str = content_mode.value if isinstance(content_mode, ContentMode) else content_mode
        logger.info(
            f"Search and fetch for '{query}' with mode: {self.search_mode}, "
            f"max_results: {max_results}, content_mode: {content_mode_str}"
        )

        try:
            # Option 1: Use tarzi's built-in search_and_fetch if using tarzi's native modes
            if self._can_use_tarzi_search_and_fetch(content_mode, fetch_mode):
                return self._search_and_fetch_tarzi_native(query, max_results, content_mode, fetch_mode, **fetch_kwargs)

            # Option 2: Search first, then fetch content using our TarziFetcher
            return self._search_and_fetch_separate(query, max_results, content_mode, **fetch_kwargs)

        except Exception as e:
            logger.error(f"Search and fetch failed for query '{query}': {e}")
            raise

    def _can_use_tarzi_search_and_fetch(self, content_mode: Union[ContentMode, str], fetch_mode: Optional[str]) -> bool:
        """
        Check if we can use tarzi's native search_and_fetch function.

        Args:
            content_mode: Content formatting mode
            fetch_mode: Fetch mode

        Returns:
            True if we can use tarzi's native function, False otherwise
        """
        # Convert string to ContentMode if needed
        if isinstance(content_mode, str):
            try:
                content_mode = ContentMode(content_mode)
            except ValueError:
                return False

        # We can use tarzi's native function for RAW_HTML and MARKDOWN modes
        return content_mode in [ContentMode.RAW_HTML, ContentMode.MARKDOWN]

    def _search_and_fetch_tarzi_native(
        self,
        query: str,
        max_results: int,
        content_mode: ContentMode,
        fetch_mode: Optional[str],
        **fetch_kwargs: Any,
    ) -> List[SearchWithContentResult]:
        """
        Use tarzi's native search_and_fetch function.

        Args:
            query: Search query
            max_results: Maximum results
            content_mode: Content mode
            fetch_mode: Fetch mode
            **fetch_kwargs: Additional fetch arguments

        Returns:
            List of SearchWithContentResult objects
        """
        import tarzi

        # Map our content modes to tarzi formats
        tarzi_format = "html" if content_mode == ContentMode.RAW_HTML else "markdown"
        tarzi_fetch_mode = fetch_mode or self._fetcher.fetch_mode

        # Use tarzi's search_and_fetch
        results_with_content = tarzi.search_and_fetch(
            query, self.search_mode, max_results, tarzi_fetch_mode, tarzi_format
        )

        # Convert to our format
        search_results = []
        for i, (result, content) in enumerate(results_with_content):
            search_result = SearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                rank=getattr(result, "rank", i + 1),
            )
            search_with_content = SearchWithContentResult(
                result=search_result, content=content, content_mode=content_mode
            )
            search_results.append(search_with_content)

        return search_results

    def _search_and_fetch_separate(
        self,
        query: str,
        max_results: int,
        content_mode: ContentMode,
        **fetch_kwargs: Any,
    ) -> List[SearchWithContentResult]:
        """
        Search first, then fetch content separately using TarziFetcher.

        Args:
            query: Search query
            max_results: Maximum results
            content_mode: Content mode
            **fetch_kwargs: Additional fetch arguments

        Returns:
            List of SearchWithContentResult objects
        """
        # First, perform the search
        search_results = self.search(query, max_results)

        # Then, fetch content for each result
        results_with_content = []
        for result in search_results:
            try:
                content = self._fetcher.fetch(result.url, content_mode, **fetch_kwargs)
                search_with_content = SearchWithContentResult(result=result, content=content, content_mode=content_mode)
                results_with_content.append(search_with_content)
            except Exception as e:
                logger.warning(f"Failed to fetch content from {result.url}: {e}")
                # Add result with empty content rather than skipping
                search_with_content = SearchWithContentResult(
                    result=result,
                    content=f"Failed to fetch content: {str(e)}",
                    content_mode=content_mode,
                )
                results_with_content.append(search_with_content)

        return results_with_content

    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported search engines.

        Returns:
            List of supported search engine names
        """
        return [
            "google",
            "bing",
            "duckduckgo",
            "brave",
            "baidu",
            "exa",
            "travily",
        ]

    def get_supported_content_modes(self) -> Dict[str, str]:
        """
        Get supported content modes for search_and_fetch.

        Returns:
            Dictionary mapping mode names to descriptions
        """
        return self._fetcher.get_supported_modes()
