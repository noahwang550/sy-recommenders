"""Tests for the recommend MCP tool."""

import pandas as pd
import pytest

from mcp_server.state import StateNotFoundError, StateStore


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return decorator


def _sar_train_df():
    return pd.DataFrame(
        {
            "userID": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "itemID": ["i1", "i2", "i2", "i3", "i1", "i3", "i4", "i1"],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 5.0, 3.0, 4.0],
            "timestamp": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )


def _register_and_get_tool():
    from mcp_server.tools.score import register_score_tools

    server = MockServer()
    register_score_tools(server)
    return server.tools["recommend"]


def test_recommend_sar_returns_payload_with_extra_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    from recommenders.models.sar.sar_singlenode import SARSingleNode

    df = _sar_train_df()
    model = SARSingleNode(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
    )
    model.fit(df)

    store = StateStore(str(tmp_path / "state"))
    handle = store.put_model(model)

    test = df[df["userID"] == "u1"][["userID", "itemID"]]
    recommend = _register_and_get_tool()
    payload = recommend(
        model_handle=handle,
        user_data=test.to_json(orient="split"),
        top_k=2,
    )

    assert "uri" in payload
    assert "rows" in payload
    assert "schema" in payload
    assert "skipped_user_count" in payload
    assert payload["model_handle"] == handle
    assert payload["rows"] > 0


def test_recommend_sar_skips_unknown_users(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    from recommenders.models.sar.sar_singlenode import SARSingleNode

    df = _sar_train_df()
    model = SARSingleNode(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
    )
    model.fit(df)

    store = StateStore(str(tmp_path / "state"))
    handle = store.put_model(model)

    test = pd.DataFrame({"userID": ["u1", "u5"], "itemID": ["i1", "i1"]})
    recommend = _register_and_get_tool()
    payload = recommend(
        model_handle=handle,
        user_data=test.to_json(orient="split"),
        top_k=2,
    )

    assert payload["skipped_user_count"] >= 1
    recs = pd.read_json(payload["uri"], orient="split")
    assert "u5" not in recs["userID"].values


def test_recommend_sar_all_unknown_users_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    from recommenders.models.sar.sar_singlenode import SARSingleNode

    df = _sar_train_df()
    model = SARSingleNode(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
    )
    model.fit(df)

    store = StateStore(str(tmp_path / "state"))
    handle = store.put_model(model)

    test = pd.DataFrame({"userID": ["u9", "u10"], "itemID": ["i1", "i1"]})
    recommend = _register_and_get_tool()
    payload = recommend(
        model_handle=handle,
        user_data=test.to_json(orient="split"),
        top_k=2,
    )
    assert payload["rows"] == 0
    assert payload["skipped_user_count"] == 2


def test_recommend_invalid_handle_raises_state_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))

    test = pd.DataFrame({"userID": ["u1"], "itemID": ["i1"]})
    recommend = _register_and_get_tool()
    with pytest.raises(StateNotFoundError):
        recommend(
            model_handle="deadbeef" * 4,
            user_data=test.to_json(orient="split"),
        )


def test_recommend_unsupported_model_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))

    class Dummy:
        pass

    store = StateStore(str(tmp_path / "state"))
    handle = store.put_model(Dummy())

    test = pd.DataFrame({"userID": ["u1"], "itemID": ["i1"]})
    recommend = _register_and_get_tool()
    with pytest.raises(ValueError, match="Unsupported model type"):
        recommend(model_handle=handle, user_data=test.to_json(orient="split"))


def test_recommend_does_not_accept_pickle(monkeypatch):
    """recommend only accepts a model_handle string; cloudpickle.load must not be an input path."""
    import cloudpickle

    from mcp_server.state import StateStore

    called = {"handle": None}

    def fake_get_model(self, handle):
        called["handle"] = handle
        return object()

    def fake_load(file, *args, **kwargs):
        raise AssertionError("cloudpickle.load should not be used as an input path")

    monkeypatch.setattr(StateStore, "get_model", fake_get_model)
    monkeypatch.setattr(cloudpickle, "load", fake_load)

    recommend = _register_and_get_tool()
    test = pd.DataFrame({"userID": ["u1"], "itemID": ["i1"]})
    with pytest.raises(ValueError, match="Unsupported model type"):
        recommend(
            model_handle="deadbeef" * 4,
            user_data=test.to_json(orient="split"),
        )
    assert called["handle"] == "deadbeef" * 4
