"""Tests for the persistent state store."""

import time

import pandas as pd
import pytest

from mcp_server.state import StateStore, StateNotFoundError, StateVersionError


def test_put_get_df_roundtrip(temp_state_root):
    store = StateStore(temp_state_root)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    handle = store.put_df(df)
    assert len(handle) == 32
    restored = store.get_df(handle)
    pd.testing.assert_frame_equal(df, restored)


def test_put_get_model_roundtrip_sar(temp_state_root):
    class FakeModel:
        def recommend_k_items(self, test, top_k=10):
            return test

    store = StateStore(temp_state_root)
    model = FakeModel()
    handle = store.put_model(model)
    restored = store.get_model(handle)
    assert isinstance(restored, FakeModel)
    assert restored.recommend_k_items("x") == "x"


def test_version_mismatch_raises(temp_state_root):
    class FakeModel:
        pass

    store = StateStore(temp_state_root)
    handle = store.put_model(FakeModel())
    with pytest.raises(StateVersionError):
        store.get_model(handle, expects_version="0.0.0")


def test_ttl_expiry_removes_handle(temp_state_root):
    store = StateStore(temp_state_root, ttl_seconds=1)
    handle = store.put_model(object())
    time.sleep(2)
    store.cleanup_expired()
    with pytest.raises(StateNotFoundError):
        store.get_model(handle)


def test_handle_is_token_hex(temp_state_root):
    store = StateStore(temp_state_root)
    handle = store.put_df(pd.DataFrame({"a": [1]}))
    assert len(handle) == 32
    assert all(c in "0123456789abcdef" for c in handle)


def test_invalid_handle_raises(temp_state_root):
    store = StateStore(temp_state_root)
    with pytest.raises(ValueError):
        store.get_df("not-a-handle")
