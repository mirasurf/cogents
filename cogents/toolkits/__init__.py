"""
Built-in toolkits for cogents.

This package contains implementations of various toolkits that provide
specific functionality for LLM-based agents.
"""

# Import all toolkit implementations to trigger registration
# Use lazy imports with error handling to avoid failing when dependencies are missing
import importlib
import logging

logger = logging.getLogger(__name__)

toolkit_modules = [
    "arxiv_toolkit",
    "audio_toolkit",
    "bash_toolkit",
    "document_toolkit",
    "file_edit_toolkit",
    "github_toolkit",
    "image_toolkit",
    "memory_toolkit",
    "python_executor_toolkit",
    "search_toolkit",
    "serper_toolkit",
    "tabular_data_toolkit",
    "thinking_toolkit",
    "user_interaction_toolkit",
    "video_toolkit",
    "wikipedia_toolkit",
]

# Attempt to import each toolkit module
for module_name in toolkit_modules:
    try:
        importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        logger.warning(f"Failed to import toolkit module {module_name}: {e}")

# Import MCP integration if available
try:
    __all__ = [
        "arxiv_toolkit",
        "audio_toolkit",
        "bash_toolkit",
        "codesnip_toolkit",
        "document_toolkit",
        "file_edit_toolkit",
        "github_toolkit",
        "image_toolkit",
        "memory_toolkit",
        "python_executor_toolkit",
        "search_toolkit",
        "serper_toolkit",
        "tabular_data_toolkit",
        "thinking_toolkit",
        "user_interaction_toolkit",
        "video_toolkit",
        "wikipedia_toolkit",
        "mcp_integration",
    ]
except ImportError:
    __all__ = [
        "arxiv_toolkit",
        "audio_toolkit",
        "bash_toolkit",
        "codesnip_toolkit",
        "document_toolkit",
        "file_edit_toolkit",
        "github_toolkit",
        "image_toolkit",
        "memory_toolkit",
        "python_executor_toolkit",
        "search_toolkit",
        "serper_toolkit",
        "tabular_data_toolkit",
        "thinking_toolkit",
        "user_interaction_toolkit",
        "video_toolkit",
        "wikipedia_toolkit",
    ]
