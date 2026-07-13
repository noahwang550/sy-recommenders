"""Tests for lazy dependency loading and structured error messages."""

import pytest

from mcp_server.deps import MissingExtraError, load_model_class


def test_missing_extra_error_message_contains_install_command():
    err = MissingExtraError("gpu", "recommenders.models.ncf.NCF", "install cuda")
    message = str(err)
    assert "gpu" in message
    assert "recommenders-ai[gpu]" in message
    assert "recommenders-mcp:gpu" in message
    assert "install cuda" in message
    assert err.extra == "gpu"
    assert err.symbol == "recommenders.models.ncf.NCF"


def test_load_model_class_missing_raises():
    with pytest.raises(MissingExtraError):
        load_model_class("recommenders.models.nonexistent.module", "Foo", "gpu")
