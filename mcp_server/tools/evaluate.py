"""Evaluation MCP tools."""

import logging

from mcp_server.deps import load_eval_api
from mcp_server.schemas import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)
from mcp_server.serialization import df_from_json

logger = logging.getLogger("recommenders-ai")


def _common_kwargs(rating_true, rating_pred, col_user, col_item, col_rating=None, col_prediction=None):
    kwargs = {
        "rating_true": rating_true,
        "rating_pred": rating_pred,
        "col_user": col_user,
        "col_item": col_item,
    }
    if col_rating is not None:
        kwargs["col_rating"] = col_rating
    if col_prediction is not None:
        kwargs["col_prediction"] = col_prediction
    return kwargs


def register_evaluate_tools(server):
    from mcp_server.http_transport import _TOOL_REGISTRY

    def eval_rating(
        rating_true: str,
        rating_pred: str,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_rating: str = DEFAULT_RATING_COL,
        col_prediction: str = DEFAULT_PREDICTION_COL,
    ) -> dict:
        """Compute rating-prediction metrics: RMSE, MAE, R-squared, explained variance."""
        t = df_from_json(rating_true)
        p = df_from_json(rating_pred)
        api = load_eval_api()
        kwargs = _common_kwargs(t, p, col_user, col_item, col_rating, col_prediction)
        result = {
            "rmse": float(api["rmse"](**kwargs)),
            "mae": float(api["mae"](**kwargs)),
            "rsquared": float(api["rsquared"](**kwargs)),
            "exp_var": float(api["exp_var"](**kwargs)),
        }
        logger.info("eval_rating result=%s", result)
        return result

    def eval_classification(
        rating_true: str,
        rating_pred: str,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_rating: str = DEFAULT_RATING_COL,
        col_prediction: str = DEFAULT_PREDICTION_COL,
    ) -> dict:
        """Compute classification metrics: AUC and logloss."""
        t = df_from_json(rating_true)
        p = df_from_json(rating_pred)
        api = load_eval_api()
        kwargs = _common_kwargs(t, p, col_user, col_item, col_rating, col_prediction)
        result = {
            "auc": float(api["auc"](**kwargs)),
            "logloss": float(api["logloss"](**kwargs)),
        }
        logger.info("eval_classification result=%s", result)
        return result

    def eval_ranking(
        rating_true: str,
        rating_pred: str,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_prediction: str = DEFAULT_PREDICTION_COL,
        k: int = 10,
    ) -> dict:
        """Compute ranking metrics: precision, recall, nDCG, MAP, r_precision."""
        t = df_from_json(rating_true)
        p = df_from_json(rating_pred)
        api = load_eval_api()
        common = {"col_user": col_user, "col_item": col_item, "col_prediction": col_prediction, "k": k}
        result = {
            "precision": float(api["precision_at_k"](t, p, **common)),
            "recall": float(api["recall_at_k"](t, p, **common)),
            "ndcg": float(api["ndcg_at_k"](t, p, **common)),
            "map": float(api["map"](t, p, **common)),
            "r_precision": float(api["r_precision_at_k"](t, p, **common)),
        }
        logger.info("eval_ranking k=%d result=%s", k, result)
        return result

    def eval_beyond_accuracy(
        train_df: str,
        reco_df: str,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
    ) -> dict:
        """Compute beyond-accuracy metrics: diversity, novelty, serendipity, coverage."""
        train = df_from_json(train_df)
        reco = df_from_json(reco_df)
        api = load_eval_api()
        common = {"col_user": col_user, "col_item": col_item}
        result = {
            "diversity": float(api["diversity"](train, reco, **common)),
            "novelty": float(api["novelty"](train, reco, **common)),
            "serendipity": float(api["serendipity"](train, reco, **common)),
            "catalog_coverage": float(api["catalog_coverage"](reco, **common)),
            "distributional_coverage": float(api["distributional_coverage"](reco, **common)),
        }
        logger.info("eval_beyond_accuracy result=%s", result)
        return result

    for name, fn in (
        ("eval_rating", eval_rating),
        ("eval_classification", eval_classification),
        ("eval_ranking", eval_ranking),
        ("eval_beyond_accuracy", eval_beyond_accuracy),
    ):
        _TOOL_REGISTRY[name] = fn
        if hasattr(server, "tool"):
            server.tool(name)(fn)
