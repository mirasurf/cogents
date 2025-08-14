"""
SeekraAgent Module

This module provides advanced research capabilities using LangGraph and LLM integration
with OpenRouter.

The researcher requires both OPENROUTER_API_KEY and GEMINI_API_KEY environment
variables. OPENROUTER_API_KEY is required for LLM functionality, and GEMINI_API_KEY
is required for real web search capabilities.
"""

from .configuration import Configuration
from .prompts import get_research_prompts
from .researcher import SeekraAgent
from .schemas import Reflection, SearchQueryList
from .state import Query, QueryState, ReflectionState, ResearchState, WebSearchState

__all__ = [
    "SeekraAgent",
    "ResearchState",
    "QueryState",
    "WebSearchState",
    "ReflectionState",
    "Query",
    "SearchQueryList",
    "Reflection",
    "Configuration",
    "get_research_prompts",
]
