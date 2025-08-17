from __future__ import annotations

from typing import Any, Dict

from .models import ExecuteResult, PlanStep, ToolSpec


class BaseAdapter:
    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        raise NotImplementedError


class LocalAdapter(BaseAdapter):
    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        return ExecuteResult(output={"echo": input_data, "tool": step.tool_id})


class HTTPAdapter(BaseAdapter):
    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        try:
            import requests  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("HTTP adapter requires 'requests' package") from e

        url = step.params.get("url") or spec.config.get("url")
        method = (step.params.get("method") or spec.config.get("method") or "post").lower()
        headers = step.params.get("headers") or spec.config.get("headers") or {"Content-Type": "application/json"}
        payload = {"input": input_data, "params": step.params}

        if not url:
            raise ValueError("HTTPAdapter: missing 'url' in step.params or spec.config")

        resp = requests.request(method, url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = (
            resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}
        )
        return ExecuteResult(output=data, metrics={"status_code": resp.status_code})


class LangChainAdapter(BaseAdapter):
    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        try:
            from langchain_core.tools import BaseTool  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("LangChain adapter requires 'langchain-core'") from e

        tool_obj: BaseTool | None = step.params.get("tool_object") or spec.runtime
        tool_name: str | None = step.params.get("tool_name") or spec.name

        if tool_obj is None and tool_name is None:
            raise ValueError("LangChainAdapter: provide 'tool_object' in params or spec.runtime, or 'tool_name'")

        if tool_obj is not None:
            result = tool_obj.invoke({**input_data, **(step.params.get("kwargs") or {})})
            return ExecuteResult(output={"result": result})

        # Lookup by name can be added if a registry of LC tools is provided externally
        raise NotImplementedError("LangChainAdapter: lookup by tool_name not implemented.")


class MCPAdapter(BaseAdapter):
    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        # Expect spec.runtime to be an MCP client with a 'call' method
        client = spec.runtime
        if client is None or not hasattr(client, "call"):
            raise ValueError("MCPAdapter: spec.runtime must be an MCP client exposing 'call(method, payload)'.")

        method = step.params.get("method") or spec.config.get("method")
        payload = step.params.get("payload") or input_data
        if not method:
            raise ValueError("MCPAdapter: missing 'method' in step.params or spec.config")

        result = client.call(method, payload)
        return ExecuteResult(output={"result": result})


class AdapterRouter:
    def __init__(self):
        self._local = LocalAdapter()
        self._http = HTTPAdapter()
        self._langchain = LangChainAdapter()
        self._mcp = MCPAdapter()

    def execute(self, spec: ToolSpec, step: PlanStep, input_data: Dict[str, Any]) -> ExecuteResult:
        tool_type = (spec.type or "").lower()
        if tool_type in ("local", "custom", ""):
            return self._local.execute(spec, step, input_data)
        if tool_type == "http":
            return self._http.execute(spec, step, input_data)
        if tool_type == "langchain":
            return self._langchain.execute(spec, step, input_data)
        if tool_type == "mcp":
            return self._mcp.execute(spec, step, input_data)
        return self._local.execute(spec, step, input_data)
