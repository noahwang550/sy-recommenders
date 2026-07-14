"""
Generic TF-IDF training entry point for user-supplied data (parquet/csv).
Source: examples/00_quick_start/tfidf_covid.ipynb (adapted for user-supplied data)
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
    p = argparse.ArgumentParser(description="TF-IDF training on a user-supplied data file")
    p.add_argument("--data", required=True, help="Path to user data file (parquet/csv/tsv)")
    p.add_argument("--col-item", default="itemID", help="Item column name")
    p.add_argument("--col-text", default="text", help="Text column name")
    p.add_argument("--top-k", type=int, default=10, help="Top-k recommendations")
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
    from recommenders.models.tfidf.tfidf_utils import TfidfRecommender

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(2)

    df = _read_dataframe(Path(args.data))
    col_item = args.col_item
    col_text = args.col_text

    if col_item not in df.columns:
        print(f"Error: item column '{col_item}' not found", file=sys.stderr)
        sys.exit(2)
    if col_text not in df.columns:
        print(f"Error: text column '{col_text}' not found", file=sys.stderr)
        sys.exit(2)

    # TfidfRecommender expects id_col positionally.
    model = TfidfRecommender(id_col=col_item)
    tf, vectors_tokenized = model.tokenize_text(
        df_clean=df, text_col=col_text, ngram_range=(1, 3), min_df=0.0
    )
    model.fit(tf, vectors_tokenized)
    topk = model.recommend_top_k_items(df, k=args.top_k)

    metrics = {
        "status": "trained",
        "n_items": df[col_item].nunique(),
        "n_rows": len(df),
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
