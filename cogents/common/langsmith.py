"""
LangSmith configuration for observability of LLM communications.

This module configures LangSmith tracing to monitor and analyze
all LLM interactions in the CogentNano application.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing based on environment variables.

    Returns:
        bool: True if LangSmith was successfully configured, False otherwise
    """
    try:
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "default")
        langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

        if not langsmith_api_key:
            return False

        if not langsmith_tracing:
            return False

        # Set up LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
        return True

    except Exception as e:
        logger.error(f"Failed to configure LangSmith: {e}")
        return False


def get_langsmith_project() -> Optional[str]:
    """
    Get the current LangSmith project name.

    Returns:
        Optional[str]: Project name if configured, None otherwise
    """
    return os.getenv("LANGCHAIN_PROJECT")


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is currently enabled.

    Returns:
        bool: True if LangSmith tracing is enabled, False otherwise
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


# Auto-configure on import if environment variables are set
if __name__ != "__main__":
    configure_langsmith()
