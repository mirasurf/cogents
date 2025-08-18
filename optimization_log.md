# Production Code Optimization Log

This file documents potential improvements to the production code that were identified while fixing unit tests.

## Issues Identified

### 1. DecomposerRegistry API Design Issue
**File:** `cogents/goalith/decomposer/registry.py`
**Issue:** The `register()` method only accepts a decomposer object and uses its intrinsic `name` property, but many tests expect to be able to specify a custom registry name. This creates a mismatch between the API design and expected usage.

**Current API:**
```python
def register(self, decomposer: GoalDecomposer, replace: bool = False) -> None:
```

**Suggested Improvement:** 
Consider adding an optional `name` parameter to allow custom registry names:
```python
def register(self, decomposer: GoalDecomposer, name: Optional[str] = None, replace: bool = False) -> None:
    registry_name = name or decomposer.name
    # ... rest of implementation
```

### 2. MemoryManager API Inconsistency
**File:** `cogents/goalith/memory/manager.py`
**Issue:** Tests expect methods like `store_context()`, `get_context()`, `store_execution_note()`, `search_similar_goals()` but the actual implementation has different method names.

**Missing Methods (based on test expectations):**
- `store_context(node_id, key, context_data)`
- `get_context(node_id, key)`
- `store_execution_note(node_id, note)`
- `search_similar_goals(goal_node, filters=None, limit=None)`

**Suggested Improvement:** 
Either implement these methods or update the tests to use the correct API. The current implementation seems to have different method names that may be more appropriate.

### 3. GraphStore Missing Methods
**File:** `cogents/goalith/base/graph_store.py`
**Issue:** Tests expect methods that don't exist in the implementation.

**Missing Methods:**
- `get_parents(node_id)`
- `list_nodes()`
- `save_graph(filename)`
- `load_graph(filename)`

**Suggested Improvement:**
Implement these utility methods or update tests to use existing API.

### 4. Scheduler Statistics Not Tracked
**File:** `cogents/goalith/schedule/scheduler.py`
**Issue:** Tests expect statistics tracking for `get_next_calls` but the implementation doesn't track this metric.

**Suggested Improvement:**
Add proper statistics tracking for all operations or remove the expectation from tests.

### 5. UpdateQueue Configuration Issue
**File:** `cogents/goalith/update/update_queue.py`
**Issue:** Tests expect default maxsize of 1000, but implementation uses 0 (unlimited). Also missing `is_full()` method.

**Suggested Improvement:**
- Set appropriate default maxsize
- Implement missing utility methods like `is_full()`

### 6. Abstract Class Test Implementation Issues
**Issue:** Several tests try to instantiate abstract classes directly instead of creating proper test implementations.

**Suggested Improvement:**
The test patterns suggest that concrete test implementations should be provided for abstract base classes to enable proper testing.

### 7. Deprecated datetime.utcnow() Usage
**File:** `cogents/goalith/memory/manager.py`
**Issue:** Code uses deprecated `datetime.utcnow()` instead of timezone-aware `datetime.now(timezone.utc)`.

**Suggested Improvement:**
Replace all instances of `datetime.utcnow()` with `datetime.now(timezone.utc)`.

## Recommendations

1. **API Consistency**: Review and standardize the API across all modules to ensure consistency between method names and expected usage patterns.

2. **Test-Driven Development**: The tests reveal expected functionality that isn't implemented. Consider implementing these features or updating the tests to match the intended API.

3. **Documentation**: Update API documentation to clearly specify the expected method signatures and behavior.

4. **Deprecation Handling**: Update deprecated datetime usage throughout the codebase.

5. **Error Handling**: Some tests expect specific error types that may not be properly implemented in the production code.