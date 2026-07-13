"""Tests for data-loading MCP tools."""

from unittest import mock

import pandas as pd
import pytest

from mcp_server.tools.data import register_data_tools


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn
        return decorator


def test_load_movielens(monkeypatch):
    server = MockServer()
    fake_df = pd.DataFrame({"userID": [1, 2], "itemID": [10, 20], "rating": [4.0, 5.0]})

    def fake_loader(size, header=None, local_cache_path=None, title_col=None, genres_col=None, year_col=None):
        return fake_df

    monkeypatch.setattr("mcp_server.tools.data.load_movielens_loader", lambda: fake_loader)
    register_data_tools(server)
    result = server.tools["load_movielens"]("100k")
    assert result["rows"] == 2
    assert result["uri"].startswith('{"')


def test_load_criteo(monkeypatch):
    server = MockServer()
    fake_df = pd.DataFrame({"userID": [1], "itemID": [10], "rating": [1.0]})

    def fake_loader(size, local_cache_path=None, header=None):
        return fake_df

    monkeypatch.setattr("mcp_server.tools.data.load_criteo_loader", lambda: fake_loader)
    register_data_tools(server)
    result = server.tools["load_criteo"]("sample")
    assert result["rows"] == 1


def test_load_mind(monkeypatch, tmp_path):
    server = MockServer()

    def fake_download(size, dest_path=None):
        return str(tmp_path / "train.zip"), str(tmp_path / "valid.zip")

    def fake_extract(train_zip, valid_zip, train_folder="train", valid_folder="valid", clean_zip_file=True):
        return str(tmp_path / "train"), str(tmp_path / "valid")

    monkeypatch.setattr("mcp_server.tools.data.load_mind_api", lambda: (fake_download, fake_extract))
    register_data_tools(server)
    result = server.tools["load_mind"]("small", str(tmp_path))
    assert result["size"] == "small"
    assert "train_path" in result
