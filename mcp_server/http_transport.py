"""ASGI/HTTP transport for the MCP server.

Exposes a minimal REST surface:
  - GET  /health
  - POST /invoke  {tool: str, arguments: dict}

All requests must carry ``Authorization: Bearer <MCP_HTTP_TOKEN>``.
The actual tool dispatch uses the in-memory ``_TOOL_REGISTRY`` populated
by ``server._register_all``.
"""

import inspect
import logging
import os
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mcp_server.auth import AuthConfigError, extract_bearer, get_expected_token, verify_token

logger = logging.getLogger("recommenders-ai")

# Populated by server._register_all().
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


class ToolRegistry:
    """Simple holder for tool functions; mirrors a subset of the MCP server API."""

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def tool(self, name: str | None = None):
        def decorator(fn: Callable[..., Any]):
            nonlocal name
            resolved = name or fn.__name__
            self._tools[resolved] = fn
            _TOOL_REGISTRY[resolved] = fn
            return fn

        return decorator

    def list_tools(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, fn in self._tools.items():
            sig = inspect.signature(fn)
            result[name] = {
                "name": name,
                "parameters": {
                    name: {
                        "type": "string" if param.annotation == str else "any"
                    }
                    for name, param in sig.parameters.items()
                },
            }
        return result

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        fn = self._tools.get(name)
        if fn is None:
            raise KeyError(f"Unknown tool: {name}")
        if inspect.iscoroutinefunction(fn):
            return await fn(**arguments)
        return fn(**arguments)


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


def build_app(server: Any | None = None) -> FastAPI:
    """Build a FastAPI application with token auth and tool dispatch."""
    expected_token = get_expected_token()

    app = FastAPI(title="recommenders-mcp")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/tools")
    async def list_tools(request: Request) -> JSONResponse:
        token = extract_bearer(request.headers.get("Authorization"))
        if not verify_token(token or "", expected_token):
            return _unauthorized()
        return JSONResponse(content={"tools": list(_TOOL_REGISTRY.keys())})

    @app.post("/invoke")
    async def invoke(request: Request) -> JSONResponse:
        token = extract_bearer(request.headers.get("Authorization"))
        if not verify_token(token or "", expected_token):
            return _unauthorized()
        body = await request.json()
        name = body.get("tool")
        arguments = body.get("arguments", {})
        if name not in _TOOL_REGISTRY:
            return JSONResponse(status_code=404, content={"error": "tool not found"})
        try:
            fn = _TOOL_REGISTRY[name]
            if inspect.iscoroutinefunction(fn):
                result = await fn(**arguments)
            else:
                result = fn(**arguments)
            return JSONResponse(content={"result": result})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool invocation failed: %s", name)
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
