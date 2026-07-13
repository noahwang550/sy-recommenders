"""Tests for split MCP tools."""

import pandas as pd
import pytest

from mcp_server.tools.split import register_split_tools


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn
        return decorator


def test_split_random_handles_list_return():
    server = MockServer()
    train_df = pd.DataFrame({"userID": [1, 2], "itemID": [10, 20]})
    test_df = pd.DataFrame({"userID": [3], "itemID": [30]})

    def fake_split(data, ratio, seed=42):
        return [train_df, test_df]

    import mcp_server.tools.split
    original = mcp_server.tools.split.load_splitters
    mcp_server.tools.split.load_splitters = lambda: {"random": fake_split}
    try:
        register_split_tools(server)
        payload = "{\"columns\":[\"userID\",\"itemID\"],\"index\":[0,1,2],\"data\":[[1,10],[2,20],[3,30]]}"
        result = server.tools["split_random"](payload, 0.75)
        assert result["train"]["rows"] == 2
        assert result["test"]["rows"] == 1
    finally:
        mcp_server.tools.split.load_splitters = original


def test_split_chrono_handles_list_return():
    server = MockServer()
    train_df = pd.DataFrame({"userID": [1], "itemID": [10], "timestamp": [1]})
    test_df = pd.DataFrame({"userID": [2], "itemID": [20], "timestamp": [2]})

    def fake_split(data, ratio, col_user, col_item, col_timestamp):
        return [train_df, test_df]

    import mcp_server.tools.split
    original = mcp_server.tools.split.load_splitters
    mcp_server.tools.split.load_splitters = lambda: {"chrono": fake_split}
    try:
        register_split_tools(server)
        payload = '{"columns":["userID","itemID","timestamp"],"index":[0,1],"data":[[1,10,1],[2,20,2]]}'
        result = server.tools["split_chrono"](payload, 0.75)
        assert result["train"]["rows"] == 1
        assert result["test"]["rows"] == 1
    finally:
        mcp_server.tools.split.load_splitters = original
