"""Tool registration helpers."""

from mcp_server.tools.data import register_data_tools
from mcp_server.tools.evaluate import register_evaluate_tools
from mcp_server.tools.ranking import register_ranking_tools
from mcp_server.tools.split import register_split_tools

__all__ = [
    "register_data_tools",
    "register_split_tools",
    "register_evaluate_tools",
    "register_ranking_tools",
]
