"""
Generic SAR training entry point for user-supplied data (parquet/csv).
Not derived from a single upstream notebook; adapts sar_movielens.py to accept an external data file.
依赖档: core
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAR training on a user-supplied data file")
    p.add_argument("--data", required=True, help="Path to user data file (parquet/csv/tsv)")
    p.add_argument("--col-user", default="userID", help="User column name")
    p.add_argument("--col-item", default="itemID", help="Item column name")
    p.add_argument("--col-rating", default="rating", help="Rating column name")
    p.add_argument("--col-timestamp", default="timestamp", help="Timestamp column name")
    p.add_argument("--ratio", type=float, default=0.75, help="Train ratio")
    p.add_argument("--top-k", type=int, default=10, help="Top-k recommendations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--similarity-type", default="jaccard", help="SAR similarity type")
    p.add_argument("--model-out", action="store_true", help="Persist fitted model to state store")
    p.add_argument("--state-root", default="./state", help="State store root directory")
    return p.parse_args(argv)


def _read_dataframe(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".parquet", ".parq"):
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(
        f"Unsupported data file extension: {suffix!r} (expected .parquet/.parq/.csv/.tsv)"
    )


def main(argv=None):
    import pandas as pd

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

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(2)

    df = _read_dataframe(Path(args.data))
    col_user = args.col_user if args.col_user is not None else DEFAULT_USER_COL
    col_item = args.col_item if args.col_item is not None else DEFAULT_ITEM_COL
    col_rating = args.col_rating if args.col_rating is not None else DEFAULT_RATING_COL
    col_timestamp = args.col_timestamp if args.col_timestamp is not None else DEFAULT_TIMESTAMP_COL

    train, test = python_random_split(df, ratio=args.ratio, seed=args.seed)

    # SAR cannot score users that are absent from the training data. For very
    # small synthetic datasets this can happen by chance; restrict the test set
    # to known users while leaving real-size datasets unchanged.
    train_users = set(train[col_user].unique())
    test = test[test[col_user].isin(train_users)]

    model = SARSingleNode(
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_timestamp=col_timestamp,
        similarity_type=args.similarity_type,
    )
    model.fit(train)
    topk = model.recommend_k_items(test, top_k=args.top_k, sort_top_k=True, remove_seen=True)

    metrics = {
        "precision": precision_at_k(
            test,
            topk,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "recall": recall_at_k(
            test,
            topk,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "ndcg": ndcg_at_k(
            test,
            topk,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction="prediction",
            k=args.top_k,
        ),
        "map": map_metric(
            test,
            topk,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction="prediction",
            k=args.top_k,
        ),
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
