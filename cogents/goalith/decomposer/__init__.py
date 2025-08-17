from .callable_decomposer import CallableDecomposer
from .human_decomposer import HumanDecomposer
from .llm_decomposer import LLMDecomposer
from .registry import DecomposerRegistry
from .simple_decomposer import SimpleListDecomposer

# Default registry instance
default_registry = DecomposerRegistry()

# Register built-in decomposers
default_registry.register(HumanDecomposer())
default_registry.register(LLMDecomposer())

__all__ = [
    "default_registry",
    "DecomposerRegistry",
    "HumanDecomposer",
    "LLMDecomposer",
    "SimpleListDecomposer",
    "CallableDecomposer",
]
