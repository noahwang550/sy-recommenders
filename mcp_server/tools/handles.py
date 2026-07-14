"""Handle management MCP tools."""

import logging
import os

from mcp_server.state import StateStore

logger = logging.getLogger("recommenders-ai")


def _state_store() -> StateStore:
    return StateStore(os.environ.get("STATE_ROOT", "./state"))


def register_handle_tools(server) -> None:
    from mcp_server.http_transport import _TOOL_REGISTRY

    def list_handles(kind: str | None = None) -> list[dict]:
        """List handles in the state store, optionally filtered by kind."""
        store = _state_store()
        return store.list_handles(kind=kind)

    def describe_handle(handle: str) -> dict:
        """Return metadata for a handle without loading its payload."""
        store = _state_store()
        return store.describe_handle(handle)

    def delete_handle(handle: str) -> dict:
        """Delete a handle from the state store."""
        store = _state_store()
        deleted = store.delete_handle(handle)
        return {"handle": handle, "deleted": deleted}

    for name, fn in (
        ("list_handles", list_handles),
        ("describe_handle", describe_handle),
        ("delete_handle", delete_handle),
    ):
        _TOOL_REGISTRY[name] = fn
        if hasattr(server, "tool"):
            server.tool(name)(fn)
