"""MCP server entry point for recommenders-ai.

Supports two transports:
  - stdio (default): launched as a child process, no network exposure.
  - http: ASGI server on ``MCP_HTTP_PORT`` (default 8080), requires
    ``MCP_HTTP_TOKEN`` for Bearer auth.

Console script::

    recommenders-mcp
"""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("recommenders-ai")

# ---------------------------------------------------------------------------
# Try to import the MCP SDK.  If it is unavailable we still provide a minimal
# server surface and a clear error at runtime instead of an import-time crash.
# ---------------------------------------------------------------------------
try:
    from mcp.server import Server as _McpServer  # type: ignore
except Exception as exc:  # pragma: no cover
    _McpServer = None  # type: ignore
    logger.debug("mcp SDK not available: %s", exc)

from mcp_server.deps import MissingExtraError
from mcp_server.http_transport import _TOOL_REGISTRY


class _FallbackServer:
    """Tiny stand-in for the real mcp.server.Server when the SDK is missing."""

    def __init__(self, name: str):
        self.name = name
        self._tools: dict[str, Any] = {}

    def tool(self, name: str | None = None):
        def decorator(fn):
            nonlocal name
            resolved = name or fn.__name__
            self._tools[resolved] = fn
            _TOOL_REGISTRY[resolved] = fn
            return fn

        return decorator

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def run(self, read, write, initialization_options=None):  # type: ignore
        raise RuntimeError(
            "The 'mcp' package is not installed. Install recommenders-ai with the required extra."
        )

    def create_initialization_options(self):
        return {}


def _get_server():
    if _McpServer is None:
        return _FallbackServer("recommenders-mcp")
    return _McpServer("recommenders-mcp")


server = _get_server()


def _register_all() -> None:
    from mcp_server.tools import (
        register_data_tools,
        register_evaluate_tools,
        register_handle_tools,
        register_ranking_tools,
        register_score_tools,
        register_split_tools,
    )

    for fn in (
        register_data_tools,
        register_split_tools,
        register_evaluate_tools,
        register_ranking_tools,
        register_score_tools,
        register_handle_tools,
    ):
        try:
            fn(server)
        except MissingExtraError:
            logger.warning("Skipping tool registration due to missing extra", exc_info=True)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("MCP_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _register_all()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        if _McpServer is None:
            raise RuntimeError(
                "MCP stdio transport requires the 'mcp' package. "
                "Install with: pip install 'recommenders-ai[dev]'"
            )
        from mcp.server.stdio import stdio_server

        async def _run():
            async with stdio_server() as (read, write):  # type: ignore
                await server.run(read, write, server.create_initialization_options())  # type: ignore

        asyncio.run(_run())
    elif transport == "http":
        from mcp_server.http_transport import build_app
        import uvicorn

        uvicorn.run(
            build_app(server),
            host="0.0.0.0",
            port=int(os.environ.get("MCP_HTTP_PORT", "8080")),
        )
    else:
        raise ValueError(f"Unknown MCP_TRANSPORT: {transport}")


if __name__ == "__main__":
    main()
