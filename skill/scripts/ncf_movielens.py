"""
Source: examples/00_quick_start/ncf_movielens.ipynb
依赖档: gpu
"""
import argparse
import json
import logging
import sys

import numpy as np

from recommenders.datasets.movielens import load_pandas_df
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    map as map_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NCF Movielens quick start")
    p.add_argument("--size", default="100k")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cache-path", default=None)
    p.add_argument("--model-out", action="store_true")
    p.add_argument("--state-root", default="./state")
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Lazy-import GPU-only symbols so that --help works in core images.
    from recommenders.models.ncf.ncf_singlenode import NCF

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_pandas_df(size=args.size)
    train, test = python_random_split(df, ratio=args.ratio, seed=42)

    # NCF cannot score users that were not present during training.
    train_users = set(train[DEFAULT_USER_COL].unique())
    test = test[test[DEFAULT_USER_COL].isin(train_users)]

    n_users = df[DEFAULT_USER_COL].nunique()
    n_items = df[DEFAULT_ITEM_COL].nunique()

    model = NCF(
        n_users=n_users,
        n_items=n_items,
        n_factors=16,
        layer_sizes=[32, 16, 8],
        n_epochs=args.epochs,
        batch_size=256,
        learning_rate=1e-3,
        verbose=0,
        seed=42,
    )
    model.fit(train)

    users = test[DEFAULT_USER_COL].unique()[:1000]
    items = test[DEFAULT_ITEM_COL].unique()
    pred = []
    for user in users:
        item_input = np.random.choice(items, size=args.top_k, replace=False)
        user_input = np.full_like(item_input, user)
        predictions = model.predict(user_input, item_input, is_list=True)
        pred.extend(zip([user] * len(item_input), item_input, predictions))

    import pandas as pd
    pred_df = pd.DataFrame(pred, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, "prediction"])

    metrics = {
        "precision": precision_at_k(test, pred_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "recall": recall_at_k(test, pred_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "ndcg": ndcg_at_k(test, pred_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
        "map": map_metric(test, pred_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_prediction="prediction", k=args.top_k),
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
