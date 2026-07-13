"""Lazy import boundary for upstream recommenders symbols.

All direct imports of the heavy `recommenders` package happen inside the
loader functions below, so the MCP server can start quickly and pay the
import cost only when a tool actually needs it.
"""

from functools import lru_cache
from typing import Any, Callable


class MissingExtraError(RuntimeError):
    """Raised when a required optional extra is not installed."""

    def __init__(self, extra: str, symbol: str, hint: str = ""):
        self.extra = extra
        self.symbol = symbol
        super().__init__(
            f"Symbol '{symbol}' requires extra '{extra}'. "
            f"Install with: pip install 'recommenders-ai[{extra}]' "
            f"or pull recommenders-mcp:{extra} image. {hint}"
        )


@lru_cache(maxsize=None)
def load_movielens_loader() -> Callable[..., Any]:
    from recommenders.datasets.movielens import load_pandas_df

    return load_pandas_df


@lru_cache(maxsize=None)
def load_criteo_loader() -> Callable[..., Any]:
    from recommenders.datasets.criteo import load_pandas_df

    return load_pandas_df


@lru_cache(maxsize=None)
def load_mind_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    from recommenders.datasets.mind import download_mind, extract_mind

    return download_mind, extract_mind


@lru_cache(maxsize=None)
def load_splitters() -> dict[str, Callable[..., Any]]:
    from recommenders.datasets.python_splitters import (
        numpy_stratified_split,
        python_chrono_split,
        python_random_split,
        python_stratified_split,
    )

    return {
        "random": python_random_split,
        "chrono": python_chrono_split,
        "stratified": python_stratified_split,
        "numpy": numpy_stratified_split,
    }


@lru_cache(maxsize=None)
def load_eval_api() -> dict[str, Callable[..., Any]]:
    from recommenders.evaluation.python_evaluation import (
        auc,
        catalog_coverage,
        distributional_coverage,
        diversity,
        exp_var,
        get_top_k_items,
        logloss,
        mae,
        map,
        map_at_k,
        ndcg_at_k,
        novelty,
        precision_at_k,
        r_precision_at_k,
        recall_at_k,
        rmse,
        rsquared,
        serendipity,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "rsquared": rsquared,
        "exp_var": exp_var,
        "auc": auc,
        "logloss": logloss,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "r_precision_at_k": r_precision_at_k,
        "ndcg_at_k": ndcg_at_k,
        "map": map,
        "map_at_k": map_at_k,
        "diversity": diversity,
        "novelty": novelty,
        "serendipity": serendipity,
        "catalog_coverage": catalog_coverage,
        "distributional_coverage": distributional_coverage,
        "get_top_k_items": get_top_k_items,
    }


def load_model_class(module_path: str, class_name: str, extra: str) -> type:
    """Import a model class by dotted path, raising MissingExtraError on failure."""
    try:
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError as e:
        raise MissingExtraError(extra, f"{module_path}.{class_name}", str(e)) from e
    except AttributeError as e:
        raise MissingExtraError(extra, f"{module_path}.{class_name}", str(e)) from e
