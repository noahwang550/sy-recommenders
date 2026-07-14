"""
Source: examples/00_quick_start/tfidf_covid.ipynb
依赖档: core
"""

import argparse
import json
import logging
import sys

from recommenders.datasets.covid_utils import load_pandas_df as load_covid19_data
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender

from mcp_server.state import StateStore

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TF-IDF COVID-19 quick start")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--col-item", default="itemID", help="Item column name")
    p.add_argument("--model-out", action="store_true")
    p.add_argument("--state-root", default="./state")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        df = load_covid19_data()
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load COVID-19 data: %s", exc)
        df = None

    if df is None:
        print(json.dumps({"error": "COVID-19 data loader not available"}))
        return 1

    col_item = args.col_item
    model = TfidfRecommender(id_col=col_item)
    model.fit(df["text"], df[col_item])
    topk = model.recommend_top_k_items(df, k=args.top_k)

    print(json.dumps({"status": "trained", "topk_shape": topk.shape}))

    if args.model_out:
        store = StateStore(args.state_root)
        handle = store.put_model(model)
        print(f"MODEL_HANDLE={handle}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
