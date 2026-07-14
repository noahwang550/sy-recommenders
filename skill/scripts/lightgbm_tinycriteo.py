"""
Source: examples/00_quick_start/lightgbm_tinycriteo.ipynb
依赖档: core
"""

import argparse
import json
import logging
import sys

from recommenders.datasets.criteo import load_pandas_df
from recommenders.datasets.python_splitters import python_random_split
from recommenders.models.lightgbm.lightgbm_utils import NumEncoder

from mcp_server.state import StateStore

logger = logging.getLogger("recommenders-ai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LightGBM Criteo quick start")
    p.add_argument("--size", default="sample")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--model-out", action="store_true")
    p.add_argument("--state-root", default="./state")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_pandas_df(size=args.size)
    train, test = python_random_split(df, ratio=args.ratio, seed=42)

    encoder = NumEncoder()
    encoder.fit(train)
    X_train = encoder.transform(train)
    y_train = train["label"].values if "label" in train.columns else train.iloc[:, -1].values

    import lightgbm as lgb

    model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=50)
    model.fit(X_train, y_train)

    print(json.dumps({"status": "trained", "train_rows": len(train), "test_rows": len(test)}))

    if args.model_out:
        store = StateStore(args.state_root)
        handle = store.put_model(model)
        print(f"MODEL_HANDLE={handle}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
