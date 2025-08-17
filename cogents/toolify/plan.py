from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .models import ExecutionPlan, PlanStep
from .registry import ToolRegistry


class PlanningEngine:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def plan(self, task: Dict[str, Any], capabilities: Optional[List[str]] = None) -> ExecutionPlan:
        plan_id = str(uuid.uuid4())
        steps: List[PlanStep] = []
        edges: List[List[str]] = []

        # Direct mapping by task tag to tool id
        direct_tool_id = task.get("context", {}).get("tool_id")
        if direct_tool_id:
            steps.append(
                PlanStep(
                    id=f"step-1",
                    tool_id=direct_tool_id,
                    params=task.get("context", {}).get("params", {}),
                )
            )
            return ExecutionPlan(plan_id=plan_id, steps=steps, edges=edges)

        # Capability-based sequential plan
        capabilities = capabilities or task.get("context", {}).get("capabilities", [])
        for idx, cap in enumerate(capabilities, start=1):
            tools = self.registry.find_by_capability(cap)
            if not tools:
                # skip missing capabilities; Orchestrix can replan later
                continue
            chosen_tool = tools[0]
            steps.append(
                PlanStep(
                    id=f"step-{idx}",
                    tool_id=chosen_tool.tool_id,
                    params={},
                )
            )
            if idx > 1:
                edges.append([f"step-{idx-1}", f"step-{idx}"])

        return ExecutionPlan(plan_id=plan_id, steps=steps, edges=edges)
