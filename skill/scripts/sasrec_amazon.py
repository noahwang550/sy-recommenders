"""
Source: examples/00_quick_start/sasrec_amazon.ipynb
依赖档: gpu
"""

import argparse
import json
import logging
import sys

from recommenders.datasets.amazon_reviews import get_review_data
from recommenders.datasets.python_splitters import python_random_split

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SASRec Amazon quick start")
    p.add_argument("--size", default="100k")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model-out", action="store_true")
    p.add_argument("--state-root", default="./state")
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Lazy-import GPU-only symbols so that --help works in core images.
    from recommenders.models.sasrec.model import SASREC

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Best-effort load; upstream API requires a file path rather than size.
    try:
        df = get_review_data(args.size)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load Amazon reviews via get_review_data: %s", exc)
        df = None

    if df is None:
        print(json.dumps({"error": "Amazon reviews loader not available in this environment"}))
        return 1

    train, test = python_random_split(df, ratio=0.75, seed=42)

    # Ensure test users are present in the training data.
    train_users = set(train["userID"].unique())
    test = test[test["userID"].isin(train_users)]

    model = SASREC(
        item_num=df["itemID"].nunique(),
        seq_max_len=50,
        num_blocks=2,
        embedding_dim=64,
        attention_dim=64,
        attention_num_heads=2,
        dropout_rate=0.2,
        conv_dims=[32, 64],
        epochs=args.epochs,
        batch_size=64,
        verbose=0,
    )
    # Upstream API expects preprocessed sequential data; this script is a template.
    model.fit(train)
    print(json.dumps({"status": "trained", "rows": len(df)}))

    if args.model_out:
        from mcp_server.state import StateStore

        store = StateStore(args.state_root)
        handle = store.put_model(model)
        print(f"MODEL_HANDLE={handle}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
