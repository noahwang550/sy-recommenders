"""HTTP token authentication for the MCP HTTP transport.

HTTP mode requires the ``MCP_HTTP_TOKEN`` environment variable to be set;
without it the server refuses to start (fail-closed).  stdio mode does not
use this module because it runs as a local child process with no network
exposure.
"""

import os
import secrets


class AuthConfigError(RuntimeError):
    """Raised when the HTTP transport is misconfigured (missing token)."""


TOKEN_ENV_VAR = "MCP_HTTP_TOKEN"


def get_expected_token() -> str:
    """Return the expected Bearer token from the environment.

    Raises
    ------
    AuthConfigError
        If ``MCP_HTTP_TOKEN`` is unset or empty.
    """
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise AuthConfigError(
            f"HTTP transport requires {TOKEN_ENV_VAR} environment variable. "
            "Set it before starting the server."
        )
    return token


def verify_token(token: str, expected: str) -> bool:
    """Constant-time compare the supplied token against the expected one."""
    if not isinstance(token, str) or not isinstance(expected, str):
        return False
    return secrets.compare_digest(token, expected)


def extract_bearer(header: str | None) -> str | None:
    """Parse an ``Authorization: Bearer <token>`` header value."""
    if not header:
        return None
    parts = header.split(" ")
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None
