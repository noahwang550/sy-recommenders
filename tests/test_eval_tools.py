"""Tests for evaluation MCP tools."""

import pandas as pd
import pytest

from mcp_server.tools.evaluate import register_evaluate_tools


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return decorator


def _payload(df):
    return df.to_json(orient="split")


def test_eval_ranking_uses_map_not_map_at_k(monkeypatch):
    server = MockServer()
    called = {}

    def fake_precision(*args, **kwargs):
        called["precision_at_k"] = True
        return 0.5

    def fake_recall(*args, **kwargs):
        called["recall_at_k"] = True
        return 0.4

    def fake_ndcg(*args, **kwargs):
        called["ndcg_at_k"] = True
        return 0.3

    def fake_map(*args, **kwargs):
        called["map"] = True
        return 0.110591

    def fake_r(*args, **kwargs):
        called["r_precision_at_k"] = True
        return 0.2

    def fake_load_eval_api():
        return {
            "precision_at_k": fake_precision,
            "recall_at_k": fake_recall,
            "ndcg_at_k": fake_ndcg,
            "map": fake_map,
            "map_at_k": lambda *a, **k: 0.999,  # should not be called
            "r_precision_at_k": fake_r,
        }

    monkeypatch.setattr("mcp_server.tools.evaluate.load_eval_api", fake_load_eval_api)
    register_evaluate_tools(server)

    true = pd.DataFrame({"userID": [1, 2], "itemID": [10, 20], "rating": [4, 5]})
    pred = pd.DataFrame({"userID": [1, 2], "itemID": [10, 20], "prediction": [3.9, 4.8]})
    result = server.tools["eval_ranking"](_payload(true), _payload(pred), k=10)

    assert "map" in called
    assert "map_at_k" not in called
    assert result["map"] == pytest.approx(0.110591)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.4
    assert result["ndcg"] == 0.3
    assert result["r_precision"] == 0.2


def test_eval_rating(monkeypatch):
    server = MockServer()

    def fake_load_eval_api():
        return {
            "rmse": lambda *a, **k: 1.0,
            "mae": lambda *a, **k: 0.8,
            "rsquared": lambda *a, **k: 0.9,
            "exp_var": lambda *a, **k: 0.85,
        }

    monkeypatch.setattr("mcp_server.tools.evaluate.load_eval_api", fake_load_eval_api)
    register_evaluate_tools(server)

    true = pd.DataFrame({"userID": [1], "itemID": [10], "rating": [4.0]})
    pred = pd.DataFrame({"userID": [1], "itemID": [10], "prediction": [3.9]})
    result = server.tools["eval_rating"](_payload(true), _payload(pred))
    assert result["rmse"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(0.8)


def test_eval_classification(monkeypatch):
    server = MockServer()

    def fake_load_eval_api():
        return {
            "auc": lambda *a, **k: 0.75,
            "logloss": lambda *a, **k: 0.5,
        }

    monkeypatch.setattr("mcp_server.tools.evaluate.load_eval_api", fake_load_eval_api)
    register_evaluate_tools(server)

    true = pd.DataFrame({"userID": [1], "itemID": [10], "rating": [1.0]})
    pred = pd.DataFrame({"userID": [1], "itemID": [10], "prediction": [0.9]})
    result = server.tools["eval_classification"](_payload(true), _payload(pred))
    assert result["auc"] == pytest.approx(0.75)
    assert result["logloss"] == pytest.approx(0.5)
