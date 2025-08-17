"""
GoalithService: DAG-based goal and to-do management system.

A hierarchical goal modeling system with flexible decomposition, reactive updates,
and extensibility for multi-agent or human-AI collaborative workflows.

## Core Components

### Graph Management
- GoalNode: Data model for goals, subgoals, and tasks
- GraphStore: DAG storage and operations using NetworkX

### Decomposition
- GoalDecomposer: Abstract interface for goal decomposition
- DecomposerRegistry: Registry for managing decomposers
- Built-in decomposers: SimpleListDecomposer, CallableDecomposer, LLMDecomposer, HumanDecomposer

### Scheduling & Prioritization
- PriorityPolicy: Abstract interface for priority policies
- Scheduler: Task scheduling with priority-based selection
- Built-in policies: SimplePriorityPolicy, DeadlinePriorityPolicy, ContextualPriorityPolicy

### Updates & Notifications
- UpdateEvent: Represents system update events
- UpdateQueue: Thread-safe event queue
- UpdateProcessor: Processes and applies updates
- Notifier: Observer pattern for event broadcasting

### Conflict Management
- ConflictDetector: Detects conflicts in updates and graph states
- ConflictResolver: Abstract interface for conflict resolution
- ConflictOrchestrator: Coordinates detection and resolution

### Replanning
- ReplanTrigger: Abstract interface for replan triggers
- Replanner: Handles replanning when triggers fire
- Built-in triggers: TaskFailureTrigger, DeadlineTrigger, PriorityChangeTrigger, ExternalSignalTrigger

### Memory Integration
- MemoryInterface: Abstract interface for memory backends
- MemoryManager: Manages context enrichment and execution history
- Built-in backends: InMemoryStore, FileSystemStore

### Main Service
- GoalithService: Unified facade for all functionality

## Extensibility

The system is designed to be highly extensible:

1. **Custom Decomposers**: Implement GoalDecomposer for domain-specific decomposition logic
2. **LLM-Enhanced Decomposition**: Use structured LLM completion for intelligent goal breakdown
3. **Custom Priority Policies**: Implement PriorityPolicy for specialized scheduling
4. **Custom Memory Backends**: Implement MemoryInterface for different storage systems
5. **Custom Conflict Resolvers**: Implement ConflictResolver for domain-specific conflict handling
6. **Custom Replan Triggers**: Implement ReplanTrigger for specialized replanning conditions
7. **Contextual Intelligence**: Leverage domain knowledge and historical patterns in decomposition

## Thread Safety

The system is designed to be thread-safe for concurrent access:
- UpdateQueue uses thread-safe queues
- Memory operations are protected with locks where necessary
- Background processing can run concurrently with API calls
"""

from .base.conflict import Conflict, ConflictResolver, ConflictSeverity, ConflictType
from .base.decomposer import GoalDecomposer

# Base classes and errors
from .base.errors import CycleDetectedError, DecompositionError, NodeNotFoundError, SchedulingError
from .base.goal_node import GoalNode, NodeStatus, NodeType
from .base.graph_store import GraphStore
from .base.memory import MemoryInterface
from .base.notification import NotificationSubscriber
from .base.priority_policy import PriorityOrder, PriorityPolicy
from .base.replanning import ReplanEvent, ReplanScope, ReplanTrigger, TriggerType
from .base.update_event import UpdateEvent

# Conflict management
from .conflict import AutomaticConflictResolver, ConflictDetector, ConflictOrchestrator, HumanConflictResolver

# Decomposition
from .decomposer import (
    CallableDecomposer,
    DecomposerRegistry,
    HumanDecomposer,
    LLMDecomposer,
    SimpleListDecomposer,
    default_registry,
)

# Memory
from .memory import FileSystemStore, InMemoryStore, MemoryManager

# Notification
from .notification import CallableSubscriber, Notifier

# Replanning
from .replan import (
    DeadlineTrigger,
    ExternalSignalTrigger,
    PriorityChangeTrigger,
    Replanner,
    TaskFailureTrigger,
    create_default_triggers,
)

# Scheduling
from .schedule import ContextualPriorityPolicy, DeadlinePriorityPolicy, Scheduler, SimplePriorityPolicy

# Main service
from .service import GoalithService

# Update system
from .update import UpdateProcessor, UpdateQueue
from .update.update_processor import UpdateType

__all__ = [
    # Base classes and errors
    "CycleDetectedError",
    "DecompositionError",
    "NodeNotFoundError",
    "SchedulingError",
    "GoalNode",
    "NodeStatus",
    "NodeType",
    "GraphStore",
    "GoalDecomposer",
    "PriorityOrder",
    "PriorityPolicy",
    "MemoryInterface",
    "Conflict",
    "ConflictResolver",
    "ConflictSeverity",
    "ConflictType",
    "ReplanEvent",
    "ReplanScope",
    "ReplanTrigger",
    "TriggerType",
    "NotificationSubscriber",
    "UpdateEvent",
    # Decomposition
    "CallableDecomposer",
    "DecomposerRegistry",
    "HumanDecomposer",
    "LLMDecomposer",
    "SimpleListDecomposer",
    "default_registry",
    # Conflict management
    "AutomaticConflictResolver",
    "ConflictDetector",
    "ConflictOrchestrator",
    "HumanConflictResolver",
    # Memory
    "FileSystemStore",
    "InMemoryStore",
    "MemoryManager",
    # Scheduling
    "ContextualPriorityPolicy",
    "DeadlinePriorityPolicy",
    "Scheduler",
    "SimplePriorityPolicy",
    # Replanning
    "DeadlineTrigger",
    "ExternalSignalTrigger",
    "PriorityChangeTrigger",
    "Replanner",
    "TaskFailureTrigger",
    "create_default_triggers",
    # Notification
    "CallableSubscriber",
    "Notifier",
    # Update system
    "UpdateProcessor",
    "UpdateQueue",
    "UpdateType",
    # Main service
    "GoalithService",
]
