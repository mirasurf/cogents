from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from cogents.goalith import GoalithService, GoalNode, NodeStatus, UpdateType
from cogents.toolify.models import ExecuteRequest
from cogents.toolify.service import ToolifyService


class OrchestrixService:
    def __init__(
        self,
        goalith: GoalithService,
        toolify: ToolifyService,
        poll_interval_sec: float = 0.2,
        max_concurrent: int = 4,
        max_retries: int = 2,
    ):
        self._goalith = goalith
        self._toolify = toolify
        self._poll_interval = poll_interval_sec
        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._leased: Set[str] = set()
        self._subscribers: List[callable] = []

    def subscribe(self, callback: callable) -> None:
        self._subscribers.append(callback)

    def _emit(self, event: Dict[str, Any]) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    def _lease(self, node_id: str) -> bool:
        if node_id in self._leased:
            return False
        if len(self._leased) >= self._max_concurrent:
            return False
        self._leased.add(node_id)
        return True

    def _release(self, node_id: str) -> None:
        self._leased.discard(node_id)

    def run_once(self, limit: int = 10) -> int:
        ready = self._goalith.get_ready_tasks(limit=limit)
        dispatched = 0
        for task in ready:
            if self._lease(task.id):
                dispatched += 1
                self._dispatch_task(task)
        return dispatched

    def _dispatch_task(self, task: GoalNode) -> None:
        # Mark in progress
        self._goalith.post_update(
            UpdateType.STATUS_CHANGE, task.id, {"new_status": NodeStatus.IN_PROGRESS}, "orchestrix"
        )
        self._goalith.process_pending_updates()

        try:
            # Build plan via Toolify
            plan = self._toolify.plan(
                {
                    "node_id": task.id,
                    "description": task.description,
                    "context": task.context or {},
                }
            )

            completed_steps: List[str] = []
            while True:
                ready_steps = plan.get_ready_steps(completed_steps)
                if not ready_steps:
                    break
                for step in ready_steps:
                    idempotency_key = f"{task.id}:{step.id}"

                    # Retry loop
                    attempt = 0
                    backoff = 0.2
                    while True:
                        attempt += 1
                        try:
                            result = self._toolify.execute(
                                ExecuteRequest(step=step, input=task.context or {}, idempotency_key=idempotency_key)
                            )
                            # Store result as context update
                            self._goalith.post_update(
                                UpdateType.CONTEXT_UPDATE,
                                task.id,
                                {"context": {"step_id": step.id, "result": result.model_dump()}},
                                "orchestrix",
                            )
                            self._goalith.process_pending_updates()
                            completed_steps.append(step.id)
                            # emit event
                            self._emit({"type": "step_completed", "node_id": task.id, "step_id": step.id})
                            break
                        except Exception:
                            if attempt > self._max_retries:
                                raise
                            time.sleep(backoff)
                            backoff = min(backoff * 2.0, 2.0)

            # Completed
            self._goalith.post_update(
                UpdateType.STATUS_CHANGE, task.id, {"new_status": NodeStatus.COMPLETED}, "orchestrix"
            )
            self._goalith.process_pending_updates()
            self._emit({"type": "task_completed", "node_id": task.id})
        except Exception as e:
            self._goalith.post_update(
                UpdateType.STATUS_CHANGE, task.id, {"new_status": NodeStatus.FAILED, "error": str(e)}, "orchestrix"
            )
            self._goalith.process_pending_updates()
            self._emit({"type": "task_failed", "node_id": task.id, "error": str(e)})
        finally:
            self._release(task.id)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self._poll_interval)
