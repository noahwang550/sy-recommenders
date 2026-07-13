"""Smoke tests for generic SAR training on user-supplied data files."""

import contextlib
import importlib.util
import io
import json
import re

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("recommenders") is None,
    reason="recommenders not installed; smoke tests require upstream package",
)


def _parse_metrics(output: str) -> dict:
    """Extract the first JSON object printed by a skill script."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output: {output!r}")
    return json.loads(match.group(0))


def _make_dataframe() -> pd.DataFrame:
    from recommenders.datasets.movielens import load_pandas_df

    return load_pandas_df(size="mock100")


@pytest.mark.notebooks
def test_sar_custom_parquet(tmp_path):
    from skill.scripts.sar_custom import main

    df = _make_dataframe()
    parquet_path = tmp_path / "input.parquet"
    df.to_parquet(parquet_path)
    state_root = tmp_path / "state"

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        main(
            [
                "--data",
                str(parquet_path),
                "--top-k",
                "5",
                "--model-out",
                "--state-root",
                str(state_root),
                "--seed",
                "42",
            ]
        )
    output = f.getvalue()

    metrics = _parse_metrics(output)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "ndcg" in metrics
    assert "map" in metrics
    assert "MODEL_HANDLE=" in output


@pytest.mark.notebooks
def test_sar_custom_column_override(tmp_path):
    from skill.scripts.sar_custom import main

    df = _make_dataframe().rename(
        columns={
            "userID": "uid",
            "itemID": "iid",
            "rating": "score",
        }
    )
    parquet_path = tmp_path / "input.parquet"
    df.to_parquet(parquet_path)
    state_root = tmp_path / "state"

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        main(
            [
                "--data",
                str(parquet_path),
                "--top-k",
                "5",
                "--model-out",
                "--state-root",
                str(state_root),
                "--seed",
                "42",
                "--col-user",
                "uid",
                "--col-item",
                "iid",
                "--col-rating",
                "score",
            ]
        )
    output = f.getvalue()

    metrics = _parse_metrics(output)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "ndcg" in metrics
    assert "map" in metrics
    assert "MODEL_HANDLE=" in output


@pytest.mark.notebooks
def test_sar_custom_csv(tmp_path):
    from skill.scripts.sar_custom import main

    df = _make_dataframe()
    csv_path = tmp_path / "input.csv"
    df.to_csv(csv_path, index=False)
    state_root = tmp_path / "state"

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        main(
            [
                "--data",
                str(csv_path),
                "--top-k",
                "5",
                "--model-out",
                "--state-root",
                str(state_root),
                "--seed",
                "42",
            ]
        )
    output = f.getvalue()

    metrics = _parse_metrics(output)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "ndcg" in metrics
    assert "map" in metrics
    assert "MODEL_HANDLE=" in output
