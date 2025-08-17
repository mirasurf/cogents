from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from .models import ToolSpec


class ToolRegistry(BaseModel):
    tools: Dict[str, ToolSpec] = {}

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def register(self, tool: ToolSpec, replace: bool = False) -> None:
        if not replace and tool.tool_id in self.tools:
            raise ValueError(f"Tool {tool.tool_id} already registered")
        self.tools[tool.tool_id] = tool

    def unregister(self, tool_id: str) -> None:
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")
        del self.tools[tool_id]

    def get(self, tool_id: str) -> ToolSpec:
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")
        return self.tools[tool_id]

    def list(
        self, capability: Optional[str] = None, tag: Optional[str] = None, type_: Optional[str] = None
    ) -> List[ToolSpec]:
        specs = list(self.tools.values())
        if capability:
            specs = [s for s in specs if any(c.name == capability for c in (s.capabilities or []))]
        if tag:
            specs = [s for s in specs if tag in (s.tags or [])]
        if type_:
            specs = [s for s in specs if s.type == type_]
        return specs

    def find_by_capability(self, capability: str) -> List[ToolSpec]:
        return self.list(capability=capability)


default_tool_registry = ToolRegistry()
