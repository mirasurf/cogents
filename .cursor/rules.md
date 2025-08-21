# Cursor AI Development Rules

## Project Overview

Cogents is a collection of essential building blocks for constructing sophisticated multi-agent systems.

### Key Modules
- `cogents.common.llm` - LLM provider abstractions with support for OpenAI, OpenRouter, Ollama, and LlamaCpp
- `cogents.agents` - Various agent implementations
- `cogents.memory` - Memory management for agents
- `cogents.sensory` - Sensory processing capabilities
- `cogents.toolify` - Tool integration framework

### LLM Providers
Supported providers in `cogents.common.llm`:
- `openai` - OpenAI API compatible services
- `openrouter` - OpenRouter API
- `ollama` - Local Ollama instances  
- `llamacpp` - Local inference with llama-cpp-python (requires LLAMACPP_MODEL_PATH)

## Development Workflow

### After Each Implementation
1. **Run unit tests**: `make test-unit` to ensure tests pass
2. **Format code**: `make format` to apply consistent formatting
3. **Check quality**: `make quality` for comprehensive code quality checks

### Quick Development Checks
- `make dev-check` - Quality + unit tests (fast feedback)
- `make full-check` - All checks + tests + build (comprehensive)
- `make ci-test` - CI test suite
- `make ci-quality` - CI quality checks

### Python Command Execution

- **Always use `poetry run` for Python commands in development**
  - Use `poetry run python script.py` instead of `python script.py`
  - Use `poetry run pytest` instead of `pytest`
  - Use `poetry run python -m module` instead of `python -m module`
  - This ensures proper dependency management and virtual environment isolation

## Examples

```bash
# ✅ Correct
poetry run python examples/tools/validate_optimization.py
poetry run pytest tests/
poetry run python -m cogents.example

# ❌ Incorrect
python examples/tools/validate_optimization.py
pytest tests/
python -m cogents.example
```

## LLM Usage Examples

```python
from cogents.common.llm import get_llm_client

# OpenAI/OpenRouter providers
client = get_llm_client(provider="openai", api_key="sk-...")
client = get_llm_client(provider="openrouter", api_key="sk-...")

# Local providers
client = get_llm_client(provider="ollama", base_url="http://localhost:11434")
client = get_llm_client(provider="llamacpp", model_path="/path/to/model.gguf")

# LlamaCpp with custom settings
client = get_llm_client(
    provider="llamacpp", 
    model_path="/path/to/model.gguf",
    n_ctx=4096,              # Context window size
    n_gpu_layers=32,         # GPU layers (-1 for all)
    instructor=True          # Enable structured output
)

# Basic chat
response = client.completion([
    {"role": "user", "content": "Hello!"}
])

# Structured output (requires instructor=True)
client = get_llm_client(provider="openai", instructor=True)
result = client.structured_completion(messages, MyPydanticModel)
```

### Environment Variables

For llamacpp provider, you can set:
- `LLAMACPP_MODEL_PATH` - Path to your GGUF model file
- `LLAMACPP_CHAT_MODEL` - Model name for logging (optional)

## Testing

- Integration tests are in `tests/integration/`
- Unit tests are in `tests/unit/`
- Use `make test-unit` to run unit tests
- Use `make test-integration` to run integration tests
- Use `make test` to run all tests
- Use `poetry run pytest tests/` to run all tests (manual)
- Use `poetry run pytest -m integration` for integration tests only (manual)
- Use `poetry run pytest -m "not slow"` to skip slow tests (manual)

## Code Quality

- Use `make format` to format code (black, isort, autoflake)
- Use `make format-check` to check formatting without changes
- Use `make lint` to run linting (flake8, mypy)
- Use `make quality` to run all quality checks
- Use `make autofix` to auto-fix code quality issues

## Rationale

Using `poetry run` ensures:
- Consistent dependency versions across development environments
- Proper virtual environment activation
- Access to all project dependencies
- Isolation from system Python packages


## Environment Variables Documentation

### Rule: Production-Only Environment Variables
When creating or updating `env.example` files, **ONLY include environment variables that are actually used in production code**.

**What to include:**
- Environment variables used in `cogents/**/*.py` files
- Variables that control runtime behavior of the application
- Configuration variables that users need to set for production deployment

**What to exclude:**
- Environment variables only used in `examples/**/*.py` files
- Variables only used in `tests/**/*.py` files
- Variables only used in `thirdparty/**/*.py` files
- Test-specific configuration variables
- Example-specific configuration variables

**How to verify:**
1. Search for `os.getenv` and `os.environ` usage in production code
2. Use pattern: `cogents/**/*.py` (exclude examples, tests, thirdparty)
3. Only document variables that appear in production code search results

**Example:**
```bash
# ✅ Include - used in production code
grep_search query="os\.getenv|os\.environ" include_pattern="cogents/**/*.py" exclude_pattern="**/tests/**|**/examples/**|**/thirdparty/**"

# ❌ Don't include - only used in examples/tests
grep_search query="os\.getenv|os\.environ" include_pattern="examples/**/*.py"
```

This ensures that `env.example` files are accurate and only show variables that users actually need to configure in production environments.
