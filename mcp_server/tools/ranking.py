"""Ranking / top-k MCP tools."""

import logging

from mcp_server.deps import load_eval_api
from mcp_server.schemas import DEFAULT_RATING_COL, DEFAULT_USER_COL
from mcp_server.serialization import df_from_json, maybe_cache

logger = logging.getLogger("recommenders-ai")


def register_ranking_tools(server):
    from mcp_server.http_transport import _TOOL_REGISTRY

    def get_top_k(
        data: str,
        col_user: str = DEFAULT_USER_COL,
        col_rating: str = DEFAULT_RATING_COL,
        k: int = 10,
        cache_path: str | None = None,
    ) -> dict:
        """Return the top-k items per user, adding a ``rank`` column."""
        df = df_from_json(data)
        api = load_eval_api()
        topk = api["get_top_k_items"](
            dataframe=df,
            col_user=col_user,
            col_rating=col_rating,
            k=k,
        )
        logger.info("get_top_k k=%d users=%d", k, topk[col_user].nunique())
        return maybe_cache(topk, cache_path)

    _TOOL_REGISTRY["get_top_k"] = get_top_k
    if hasattr(server, "tool"):
        server.tool("get_top_k")(get_top_k)
