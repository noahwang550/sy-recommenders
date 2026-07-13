"""Tests for ranking MCP tools."""

import pandas as pd

from mcp_server.tools.ranking import register_ranking_tools


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn
        return decorator


def test_get_top_k_adds_rank(monkeypatch):
    server = MockServer()
    df = pd.DataFrame({
        "userID": [1, 1, 2, 2],
        "itemID": [10, 11, 20, 21],
        "rating": [5.0, 3.0, 4.0, 2.0],
    })

    def fake_get_top_k_items(dataframe, col_user, col_rating, k):
        top = dataframe.groupby(col_user).head(k).copy()
        top["rank"] = top.groupby(col_user).cumcount() + 1
        return top

    def fake_load_eval_api():
        return {"get_top_k_items": fake_get_top_k_items}

    monkeypatch.setattr("mcp_server.tools.ranking.load_eval_api", fake_load_eval_api)
    register_ranking_tools(server)

    payload = df.to_json(orient="split")
    result = server.tools["get_top_k"](payload, k=2)
    assert result["rows"] == 4
    assert "rank" in result["schema"]
