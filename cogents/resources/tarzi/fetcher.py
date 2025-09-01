"""
Tarzi-based web content fetcher with multiple formatting modes.

Provides web content fetching capabilities using the tarzi library with support for:
- Raw HTML content from tarzi
- Markdown content formatted by tarzi
- Raw content from tarzi then formatted by cogents LLM provider
"""

from enum import Enum
from typing import Any, Dict, Optional

from cogents.common.llm import get_llm_client
from cogents.common.logging import get_logger

logger = get_logger(__name__)


class ContentMode(Enum):
    """Content formatting modes for fetched web content."""

    RAW_HTML = "raw_html"  # Raw HTML content from tarzi
    MARKDOWN = "markdown"  # Markdown content formatted by tarzi
    LLM_FORMATTED = "llm_formatted"  # Raw content formatted by LLM


class TarziFetcher:
    """
    Web content fetcher using tarzi library with multiple formatting options.

    Supports three content modes:
    1. RAW_HTML: Returns raw HTML content directly from tarzi
    2. MARKDOWN: Returns markdown-formatted content from tarzi
    3. LLM_FORMATTED: Gets raw HTML from tarzi, then formats using LLM (default: llamacpp)
    """

    def __init__(
        self,
        llm_provider: str = "llamacpp",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        fetch_mode: str = "browser_headless",
        timeout: int = 30,
    ):
        """
        Initialize the TarziFetcher.

        Args:
            llm_provider: LLM provider for content formatting ("llamacpp", "openai", "ollama", etc.)
            llm_api_key: API key for the LLM provider (optional, can use env vars)
            llm_base_url: Base URL for the LLM provider (optional)
            fetch_mode: Tarzi fetch mode ("plain_request", "browser_headless", "browser_headed")
            timeout: Request timeout in seconds
        """
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.fetch_mode = fetch_mode
        self.timeout = timeout

        # Initialize LLM client for LLM_FORMATTED mode
        self._llm_client = None
        self._setup_llm_client()

        # Initialize tarzi components
        self._setup_tarzi()

    def _setup_llm_client(self) -> None:
        """Setup LLM client for content formatting."""
        try:
            if self.llm_provider:
                if self.llm_provider == "llamacpp":
                    self._llm_client = get_llm_client(
                        provider=self.llm_provider,
                    )
                    logger.info(f"Initialized LLM client with provider: {self.llm_provider}")
                else:
                    # For other providers, use api_key
                    if self.llm_api_key:
                        self._llm_client = get_llm_client(
                            provider=self.llm_provider,
                            api_key=self.llm_api_key,
                            base_url=self.llm_base_url,
                        )
                        logger.info(f"Initialized LLM client with provider: {self.llm_provider}")
                    else:
                        logger.warning("LLM API key not configured - LLM_FORMATTED mode will be unavailable")
            else:
                logger.warning("LLM provider not configured - LLM_FORMATTED mode will be unavailable")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self._llm_client = None

    def _setup_tarzi(self) -> None:
        """Setup tarzi components with configuration."""
        try:
            # Import tarzi (lazy import to avoid dependency issues)
            import tarzi

            # Create tarzi configuration
            config_str = f"""
[fetcher]
timeout = {self.timeout}
format = "html"
"""
            self._config = tarzi.Config.from_str(config_str)
            self._web_fetcher = tarzi.WebFetcher.from_config(self._config)
            self._converter = tarzi.Converter()

            logger.info("Initialized tarzi components successfully")
        except ImportError as e:
            logger.error(f"Failed to import tarzi: {e}")
            raise ImportError("tarzi library is required but not available") from e
        except Exception as e:
            logger.error(f"Failed to setup tarzi: {e}")
            raise

    def fetch(
        self,
        url: str,
        content_mode: ContentMode = ContentMode.LLM_FORMATTED,
        **kwargs: Any,
    ) -> str:
        """
        Fetch web content with specified formatting mode.

        Args:
            url: URL to fetch content from
            content_mode: Content formatting mode
            **kwargs: Additional arguments for specific modes

        Returns:
            Formatted web content as string

        Raises:
            ValueError: If content_mode is invalid or configuration is missing
            Exception: If fetching fails
        """
        if not isinstance(content_mode, ContentMode):
            # Allow string values for backward compatibility
            if isinstance(content_mode, str):
                try:
                    content_mode = ContentMode(content_mode)
                except ValueError:
                    raise ValueError(f"Invalid content_mode: {content_mode}")
            else:
                raise ValueError(f"content_mode must be ContentMode enum or string, got {type(content_mode)}")

        logger.info(f"Fetching content from {url} with mode: {content_mode.value}")

        try:
            if content_mode == ContentMode.RAW_HTML:
                return self._fetch_raw_html(url, **kwargs)
            elif content_mode == ContentMode.MARKDOWN:
                return self._fetch_markdown(url, **kwargs)
            elif content_mode == ContentMode.LLM_FORMATTED:
                return self._fetch_llm_formatted(url, **kwargs)
            else:
                raise ValueError(f"Unsupported content mode: {content_mode}")

        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            raise

    def _fetch_raw_html(self, url: str, **kwargs: Any) -> str:
        """
        Fetch raw HTML content using tarzi.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments (unused)

        Returns:
            Raw HTML content
        """
        return self._web_fetcher.fetch(url, self.fetch_mode, "html")

    def _fetch_markdown(self, url: str, **kwargs: Any) -> str:
        """
        Fetch content and convert to markdown using tarzi.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments (unused)

        Returns:
            Markdown-formatted content
        """
        return self._web_fetcher.fetch(url, self.fetch_mode, "markdown")

    def _fetch_llm_formatted(self, url: str, format_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """
        Fetch raw content and format using LLM.

        Args:
            url: URL to fetch
            format_prompt: Custom formatting prompt for the LLM
            **kwargs: Additional arguments (unused)

        Returns:
            LLM-formatted content
        """
        if not self._llm_client:
            raise ValueError("LLM client not configured - cannot use LLM_FORMATTED mode")

        # Get raw HTML content
        raw_html = self._fetch_raw_html(url)

        # Convert to clean text first
        clean_content = self._converter.convert(raw_html, "markdown")

        # Use LLM to format the content
        default_prompt = """Please clean up and format the following web content in a readable way.
Remove navigation menus, advertisements, and irrelevant elements.
Focus on the main content and present it in a clear, structured format.

Content:
{content}

Please provide a clean, well-structured version of the main content:"""

        prompt = format_prompt or default_prompt
        full_prompt = prompt.format(content=clean_content[:8000])  # Limit content size

        try:
            response = self._llm_client.completion(
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            # Fallback to markdown if LLM fails
            logger.warning("Falling back to markdown format due to LLM failure")
            return clean_content

    def fetch_raw(
        self,
        url: str,
        fetch_mode: Optional[str] = None,
    ) -> str:
        """
        Fetch raw content using tarzi without any formatting.

        Args:
            url: URL to fetch
            fetch_mode: Override default fetch mode

        Returns:
            Raw content as returned by tarzi
        """
        mode = fetch_mode or self.fetch_mode
        return self._web_fetcher.fetch_raw(url, mode)

    def get_supported_modes(self) -> Dict[str, str]:
        """
        Get supported content modes and their descriptions.

        Returns:
            Dictionary mapping mode names to descriptions
        """
        return {
            ContentMode.RAW_HTML.value: "Raw HTML content from tarzi",
            ContentMode.MARKDOWN.value: "Markdown content formatted by tarzi",
            ContentMode.LLM_FORMATTED.value: "Raw content formatted by LLM provider",
        }
