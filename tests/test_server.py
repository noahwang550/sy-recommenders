"""Tests for MCP server registration and lifecycle."""

import pytest

from mcp_server.server import _register_all, _FallbackServer


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            resolved = name or fn.__name__
            self.tools[resolved] = fn
            return fn

        return decorator


def test_server_registers_sixteen_tools():
    server = MockServer()
    _register_all_with_server(server)
    assert len(server.tools) == 16
    expected = {
        "load_movielens",
        "load_criteo",
        "load_mind",
        "split_random",
        "split_chrono",
        "split_stratified",
        "split_numpy",
        "eval_rating",
        "eval_classification",
        "eval_ranking",
        "eval_beyond_accuracy",
        "get_top_k",
        "recommend",
        "list_handles",
        "describe_handle",
        "delete_handle",
    }
    assert set(server.tools.keys()) == expected


def test_main_unknown_transport_raises(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "foo")
    from mcp_server.server import main

    with pytest.raises(ValueError, match="Unknown MCP_TRANSPORT"):
        main()


def _register_all_with_server(server):
    """Helper to register tools onto a specific server instance."""
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
        fn(server)
