"""
Tarzi web fetching and search integration for Cogents.

This module provides web content fetching and search functionality using the tarzi library,
with support for multiple content formats and LLM-powered content processing.
"""

from .fetcher import TarziFetcher
from .searcher import TarziSearcher

__all__ = ["TarziFetcher", "TarziSearcher"]
