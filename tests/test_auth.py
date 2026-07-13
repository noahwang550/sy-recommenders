"""Tests for HTTP token authentication."""

import os

import pytest

from mcp_server.auth import (
    AuthConfigError,
    extract_bearer,
    get_expected_token,
    verify_token,
)


def test_verify_token_correct_returns_true():
    assert verify_token("secret", "secret") is True


def test_verify_token_wrong_returns_false():
    assert verify_token("wrong", "secret") is False


def test_verify_token_non_string_returns_false():
    assert verify_token(None, "secret") is False
    assert verify_token("secret", None) is False


def test_missing_env_raises_auth_config_error(monkeypatch):
    monkeypatch.delenv("MCP_HTTP_TOKEN", raising=False)
    with pytest.raises(AuthConfigError):
        get_expected_token()


def test_extract_bearer_parses_header():
    assert extract_bearer("Bearer secret") == "secret"
    assert extract_bearer("Basic secret") is None
    assert extract_bearer(None) is None
