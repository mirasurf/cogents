from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class ToolCapability(BaseModel):
    name: str = Field(description="Capability identifier, e.g., 'web.search', 'nlp.summarize'")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCard(BaseModel):
    tool_id: str = Field(description="Unique tool identifier")
    name: str = Field(description="Human-readable name")
    type: Literal["langchain", "mcp", "http", "local", "custom"] = Field(
        default="custom", description="Adapter type used for execution"
    )
    version: Optional[str] = Field(default=None)
    capabilities: List[ToolCapability] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)
    runtime: Optional[Any] = Field(default=None, description="Optional runtime object for adapter (tool, client, etc.)")

    model_config = {
        "arbitrary_types_allowed": True,
    }


class PlanStep(BaseModel):
    id: str = Field(description="Stable step id for idempotency and retries")
    tool_id: str = Field(description="Tool to execute in this step")
    params: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list, description="Names of output artifacts produced by this step")


class ExecutionPlan(BaseModel):
    plan_id: str = Field(description="Unique id of the plan")
    steps: List[PlanStep] = Field(default_factory=list)
    edges: List[List[str]] = Field(default_factory=list, description="Optional DAG edges: [[from_step_id, to_step_id]]")
    artifacts: Dict[str, Any] = Field(default_factory=dict)

    def get_ready_steps(self, completed_step_ids: Sequence[str]) -> List[PlanStep]:
        if not self.edges:
            # If no edges, execute steps sequentially
            for step in self.steps:
                if step.id not in completed_step_ids:
                    return [step]
            return []
        # DAG readiness: step is ready if all its predecessors are completed
        predecessors: Dict[str, List[str]] = {}
        for src, dst in self.edges:
            predecessors.setdefault(dst, []).append(src)
            predecessors.setdefault(src, [])
        ready: List[PlanStep] = []
        completed = set(completed_step_ids)
        for step in self.steps:
            preds = predecessors.get(step.id, [])
            if all(p in completed for p in preds) and step.id not in completed:
                ready.append(step)
        return ready


class ExecuteRequest(BaseModel):
    step: PlanStep
    input: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None


class ExecuteResult(BaseModel):
    output: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class TelemetryEvent(BaseModel):
    event_type: Literal[
        "step_started",
        "step_succeeded",
        "step_failed",
        "plan_created",
        "tool_registered",
    ]
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
