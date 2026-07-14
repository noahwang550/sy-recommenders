"""Tests for handle-management MCP tools."""

import time

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


def _register_and_get_tools():
    from mcp_server.tools.handles import register_handle_tools

    server = MockServer()
    register_handle_tools(server)
    return (
        server.tools["list_handles"],
        server.tools["describe_handle"],
        server.tools["delete_handle"],
    )


def test_list_handles_returns_all(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    store = StateStore(str(tmp_path / "state"))
    store.put_df(pd.DataFrame({"a": [1, 2]}))
    store.put_df(pd.DataFrame({"a": [3, 4]}))
    store.put_model(object())

    list_handles, _, _ = _register_and_get_tools()
    all_handles = list_handles()
    assert len(all_handles) == 3

    model_handles = list_handles(kind="model")
    assert len(model_handles) == 1


def test_list_handles_skips_expired_in_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    store = StateStore(str(tmp_path / "state"), ttl_seconds=1)
    store.put_model(object())
    time.sleep(2)

    list_handles, _, _ = _register_and_get_tools()
    handles = list_handles()
    assert handles == []


def test_describe_handle_reads_meta_no_version_check(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    import cloudpickle

    store = StateStore(str(tmp_path / "state"))
    handle = store.put_model(object())

    original_load = cloudpickle.load

    def exploding_load(file, *args, **kwargs):
        raise AssertionError("describe_handle must not load the pickle")

    monkeypatch.setattr(cloudpickle, "load", exploding_load)

    _, describe_handle, _ = _register_and_get_tools()
    meta = describe_handle(handle)

    assert meta["handle"] == handle
    assert meta["kind"] == "model"
    assert "size_bytes" in meta
    assert "recommends_version" in meta


def test_describe_handle_invalid_handle_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    _, describe_handle, _ = _register_and_get_tools()
    with pytest.raises(ValueError):
        describe_handle("bad")


def test_delete_handle_removes_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    store = StateStore(str(tmp_path / "state"))
    handle = store.put_df(pd.DataFrame({"a": [1]}))

    _, _, delete_handle = _register_and_get_tools()
    result = delete_handle(handle)
    assert result == {"handle": handle, "deleted": True}
    assert store.exists(handle) is False


def test_delete_handle_nonexistent_returns_deleted_false(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    _, _, delete_handle = _register_and_get_tools()
    result = delete_handle("0" * 32)
    assert result == {"handle": "0" * 32, "deleted": False}
