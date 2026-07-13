"""Tests for the HTTP transport."""

import os

import pytest

from mcp_server.auth import AuthConfigError
from mcp_server.http_transport import build_app


def test_http_no_token_returns_401(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TOKEN", "test-token")
    from fastapi.testclient import TestClient

    app = build_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    response = client.get("/tools")
    assert response.status_code == 401


def test_http_correct_token_returns_200(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TOKEN", "test-token")
    from fastapi.testclient import TestClient

    app = build_app()
    client = TestClient(app)
    response = client.get("/tools", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200


def test_http_invoke_tool_with_token(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TOKEN", "test-token")
    from mcp_server.http_transport import _TOOL_REGISTRY

    _TOOL_REGISTRY["echo"] = lambda x: x
    from fastapi.testclient import TestClient

    app = build_app()
    client = TestClient(app)
    response = client.post(
        "/invoke",
        headers={"Authorization": "Bearer test-token"},
        json={"tool": "echo", "arguments": {"x": 42}},
    )
    assert response.status_code == 200
    assert response.json() == {"result": 42}


def test_build_app_missing_env_raises(monkeypatch):
    monkeypatch.delenv("MCP_HTTP_TOKEN", raising=False)
    with pytest.raises(AuthConfigError):
        build_app()
