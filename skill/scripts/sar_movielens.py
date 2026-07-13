"""
Source: examples/00_quick_start/sar_movielens.ipynb
依赖档: core
"""
import argparse
import json
import logging
import sys

from recommenders.datasets.movielens import load_pandas_df
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map as map_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.sar.sar_singlenode import SARSingleNode
from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAR Movielens quick start")
    p.add_argument("--size", default="100k", help="Movielens size")
    p.add_argument("--ratio", type=float, default=0.75, help="Train ratio")
    p.add_argument("--top-k", type=int, default=10, help="Top-k recommendations")
    p.add_argument("--cache-path", default=None, help="Directory for DataFrame caching")
    p.add_argument("--model-out", action="store_true", help="Persist fitted model to state store")
    p.add_argument("--state-root", default="./state", help="State store root directory")
    p.add_argument("--similarity-type", default="jaccard", help="SAR similarity type")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_pandas_df(size=args.size)
    train, test = python_random_split(df, ratio=args.ratio, seed=42)

    # SAR cannot score users that are absent from the training data. For very
    # small synthetic datasets this can happen by chance; restrict the test set
    # to known users while leaving real-size datasets unchanged.
    train_users = set(train[DEFAULT_USER_COL].unique())
    test = test[test[DEFAULT_USER_COL].isin(train_users)]

    model = SARSingleNode(
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        similarity_type=args.similarity_type,
    )
    model.fit(train)
    topk = model.recommend_k_items(test, top_k=args.top_k, sort_top_k=True, remove_seen=True)

    metrics = {
        "precision": precision_at_k(test, topk, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "recall": recall_at_k(test, topk, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "ndcg": ndcg_at_k(test, topk, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "map": map_metric(test, topk, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
    }
    print(json.dumps(metrics, indent=2))

    if args.model_out:
        from mcp_server.state import StateStore
        store = StateStore(args.state_root)
        handle = store.put_model(model)
        print(f"MODEL_HANDLE={handle}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
