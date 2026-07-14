"""Tests for typed error envelope mapping."""

import pytest

from mcp_server.deps import MissingExtraError
from mcp_server.errors import to_response
from mcp_server.state import StateNotFoundError, StateVersionError


def test_state_not_found_maps_410():
    exc = StateNotFoundError("deadbeef" * 4)
    status, body = to_response(exc)
    assert status == 410
    assert body["code"] == "state_not_found"
    assert body["error"] == str(exc)
    assert body["details"]["handle"] == "deadbeef" * 4


def test_state_version_error_maps_409():
    exc = StateVersionError(expected="1.2.1", found="1.2.0")
    status, body = to_response(exc)
    assert status == 409
    assert body["code"] == "state_version_mismatch"
    assert body["details"]["expected"] == "1.2.1"
    assert body["details"]["found"] == "1.2.0"


def test_missing_extra_maps_503():
    exc = MissingExtraError(extra="gpu", symbol="recommenders.models.ncf.NCF", hint="install cuda")
    status, body = to_response(exc)
    assert status == 503
    assert body["code"] == "missing_extra"
    assert body["details"]["extra"] == "gpu"
    assert body["details"]["symbol"] == "recommenders.models.ncf.NCF"


def test_value_error_maps_400():
    exc = ValueError("bad request payload")
    status, body = to_response(exc)
    assert status == 400
    assert body["code"] == "bad_request"


def test_state_version_error_checked_before_value_error():
    """StateVersionError subclasses ValueError, so it must be caught first."""
    exc = StateVersionError(expected="1.2.1", found="1.2.0")
    status, body = to_response(exc)
    assert status == 409
    assert body["code"] == "state_version_mismatch"


def test_unknown_exception_maps_500():
    exc = RuntimeError("something bad")
    status, body = to_response(exc)
    assert status == 500
    assert body["code"] == "internal_error"
    assert body["details"] == {}


def test_error_field_always_str_exc():
    """Backward-compat: the 'error' key stays str(exc)."""
    exc = ValueError("msg")
    _, body = to_response(exc)
    assert body["error"] == "msg"


def test_http_invoke_state_not_found_returns_410(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_TOKEN", "test-token")
    from fastapi.testclient import TestClient
    from mcp_server.http_transport import _TOOL_REGISTRY, build_app

    def _tool():
        raise StateNotFoundError("deadbeef" * 4)

    _TOOL_REGISTRY["fake_missing"] = _tool
    app = build_app()
    client = TestClient(app)
    response = client.post(
        "/invoke",
        headers={"Authorization": "Bearer test-token"},
        json={"tool": "fake_missing", "arguments": {}},
    )
    assert response.status_code == 410
    assert response.json()["code"] == "state_not_found"
