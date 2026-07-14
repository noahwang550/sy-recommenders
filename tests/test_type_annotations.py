"""Smoke tests for type annotations on private helpers flagged by code review."""

import typing

from mcp_server.tools import handles, score


class MockServer:
    def __init__(self):
        self.tools = {}

    def tool(self, name=None):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return decorator


def test_score_helpers_have_model_annotation():
    """Duck-typed model param should be annotated (as Any) so it is documented."""
    assert "model" in typing.get_type_hints(score._sar_known_users)
    assert "model" in typing.get_type_hints(score._score_sar)
    assert "model" in typing.get_type_hints(score._score_tfidf)
    assert "model" in typing.get_type_hints(score._score_with_model)
    assert "server" in typing.get_type_hints(score.register_score_tools)


def test_handle_helpers_have_server_annotation():
    """Server param should be annotated (as Any) so it is documented."""
    assert "server" in typing.get_type_hints(handles.register_handle_tools)


def test_score_and_handles_modules_import_and_register():
    """Annotation-only changes must not affect runtime import/registration."""
    score_server = MockServer()
    handles_server = MockServer()

    score.register_score_tools(score_server)
    handles.register_handle_tools(handles_server)

    assert "recommend" in score_server.tools
    assert "list_handles" in handles_server.tools
    assert "describe_handle" in handles_server.tools
    assert "delete_handle" in handles_server.tools
