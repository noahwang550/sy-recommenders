"""Typed error envelope for MCP tool failures.

Maps domain exceptions to HTTP status codes and a structured
``{"error": str, "code": str, "details": dict}`` body.  ``error`` is kept
as ``str(exc)`` for backward compatibility; ``code`` and ``details`` are
additive and do not alter existing success-path contracts.
"""

from typing import Any

from mcp_server.deps import MissingExtraError
from mcp_server.state import StateNotFoundError, StateVersionError

# Ordered most-specific first. StateVersionError subclasses ValueError,
# so it MUST be checked before ValueError.
_EXC_MAP = (
    (StateNotFoundError, 410, "state_not_found"),
    (StateVersionError, 409, "state_version_mismatch"),
    (MissingExtraError, 503, "missing_extra"),
    (ValueError, 400, "bad_request"),
    (TypeError, 400, "bad_request"),
)


def to_response(exc: BaseException) -> tuple[int, dict[str, Any]]:
    for exc_type, status, code in _EXC_MAP:
        if isinstance(exc, exc_type):
            return status, {"error": str(exc), "code": code, "details": _extract_details(exc)}
    return 500, {"error": str(exc), "code": "internal_error", "details": {}}


def _extract_details(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, MissingExtraError):
        return {"extra": exc.extra, "symbol": exc.symbol}
    if isinstance(exc, StateVersionError):
        return {"expected": exc.expected, "found": exc.found}
    if isinstance(exc, StateNotFoundError):
        handle = exc.args[0] if exc.args else ""
        return {"handle": str(handle)}
    return {}
