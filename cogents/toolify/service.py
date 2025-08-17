from __future__ import annotations

from typing import Any, Dict, List, Optional

from .execute import AdapterRouter
from .models import ExecuteRequest, ExecuteResult, ExecutionPlan, TelemetryEvent, ToolSpec
from .plan import PlanningEngine
from .registry import ToolRegistry, default_tool_registry


class ToolifyService:
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or default_tool_registry
        self._planner = PlanningEngine(self._registry)
        self._router = AdapterRouter()
        self._subscribers: List[callable] = []
        self._idempotent_cache: Dict[str, ExecuteResult] = {}

    # Registry APIs
    def register_tool(self, spec: ToolSpec, replace: bool = False) -> None:
        self._registry.register(spec, replace=replace)
        self._emit(TelemetryEvent(event_type="tool_registered", data={"tool_id": spec.tool_id}).model_dump())

    def list_tools(self) -> List[ToolSpec]:
        return list(self._registry.tools.values())

    # Planning
    def plan(self, task: Dict[str, Any], capabilities: Optional[List[str]] = None) -> ExecutionPlan:
        plan = self._planner.plan(task, capabilities)
        self._emit(TelemetryEvent(event_type="plan_created", data={"plan_id": plan.plan_id}).model_dump())
        return plan

    # Execution
    def execute(self, request: ExecuteRequest) -> ExecuteResult:
        # Idempotency check
        if request.idempotency_key and request.idempotency_key in self._idempotent_cache:
            return self._idempotent_cache[request.idempotency_key]

        # Resolve tool spec
        spec = self._registry.get(request.step.tool_id)

        self._emit(TelemetryEvent(event_type="step_started", data={"step_id": request.step.id}).model_dump())
        try:
            result = self._router.execute(spec, request.step, request.input)
            if request.idempotency_key:
                self._idempotent_cache[request.idempotency_key] = result
            self._emit(TelemetryEvent(event_type="step_succeeded", data={"step_id": request.step.id}).model_dump())
            return result
        except Exception as e:
            self._emit(
                TelemetryEvent(
                    event_type="step_failed", data={"step_id": request.step.id, "error": str(e)}
                ).model_dump()
            )
            raise

    # Telemetry (simple observer)
    def subscribe(self, callback: callable) -> None:
        self._subscribers.append(callback)

    def _emit(self, event: Dict[str, Any]) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass
