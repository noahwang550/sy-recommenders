"""Score / recommend MCP tools."""

import logging
import os
from typing import Any

import pandas as pd

from mcp_server.schemas import DEFAULT_USER_COL
from mcp_server.serialization import df_from_json, maybe_cache
from mcp_server.state import StateStore

logger = logging.getLogger("recommenders-ai")


def _sar_known_users(model: Any) -> set:
    user2index = getattr(model, "user2index", None)
    if user2index is None:
        raise ValueError("Model is not fitted; SAR user2index is None")
    return set(user2index.keys())


def _score_sar(
    model: Any,
    user_df: pd.DataFrame,
    top_k: int,
    col_user: str,
    remove_seen: bool,
) -> tuple[pd.DataFrame, int]:
    known = _sar_known_users(model)
    mask = user_df[col_user].isin(known)
    skipped = int((~mask).sum())
    scoreable = user_df[mask]
    if scoreable.empty:
        return pd.DataFrame(), skipped
    recs = model.recommend_k_items(scoreable, top_k=top_k, sort_top_k=True, remove_seen=remove_seen)
    return recs, skipped


def _score_tfidf(model: Any, user_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, int]:
    recs = model.recommend_top_k_items(user_df, k=top_k)
    return recs, 0


def _score_with_model(
    model: Any,
    user_df: pd.DataFrame,
    top_k: int,
    col_user: str,
    remove_seen: bool,
) -> tuple[pd.DataFrame, int]:
    if hasattr(model, "recommend_k_items"):
        return _score_sar(model, user_df, top_k, col_user, remove_seen)
    if hasattr(model, "recommend_top_k_items"):
        return _score_tfidf(model, user_df, top_k)
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def register_score_tools(server: Any) -> None:
    from mcp_server.http_transport import _TOOL_REGISTRY

    def recommend(
        model_handle: str,
        user_data: str,
        top_k: int = 10,
        col_user: str = DEFAULT_USER_COL,
        remove_seen: bool = True,
        cache_path: str | None = None,
    ) -> dict:
        """Score users against a persisted model and return recommendations."""
        store = StateStore(os.environ.get("STATE_ROOT", "./state"))
        model = store.get_model(model_handle)
        user_df = df_from_json(user_data)
        recs, skipped = _score_with_model(model, user_df, top_k, col_user, remove_seen)
        payload = maybe_cache(recs, cache_path)
        return {**payload, "skipped_user_count": skipped, "model_handle": model_handle}

    _TOOL_REGISTRY["recommend"] = recommend
    if hasattr(server, "tool"):
        server.tool("recommend")(recommend)
