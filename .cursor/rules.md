# Cursor AI Development Rules

## Python Command Execution

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

## Rationale

Using `poetry run` ensures:
- Consistent dependency versions across development environments
- Proper virtual environment activation
- Access to all project dependencies
- Isolation from system Python packages