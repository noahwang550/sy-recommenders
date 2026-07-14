"""
Source: examples/00_quick_start/eval_quickstart.ipynb
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
from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generic SAR Movielens eval quickstart")
    p.add_argument("--size", default="mock100")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_pandas_df(size=args.size)
    train, test = python_random_split(df, ratio=args.ratio, seed=42)

    # Keep only test users seen during training so SAR can score them.
    train_users = set(train[DEFAULT_USER_COL].unique())
    test = test[test[DEFAULT_USER_COL].isin(train_users)]

    model = SARSingleNode(
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        similarity_type="jaccard",
    )
    model.fit(train)
    topk = model.recommend_k_items(test, top_k=args.top_k, sort_top_k=True, remove_seen=True)

    metrics = {
        "precision": precision_at_k(
            test,
            topk,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "recall": recall_at_k(
            test,
            topk,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "ndcg": ndcg_at_k(
            test,
            topk,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "map": map_metric(
            test,
            topk,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_prediction="prediction",
            k=args.top_k,
        ),
    }
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
