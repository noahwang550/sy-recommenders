"""Smoke tests for generic TF-IDF training on user-supplied data files."""

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
    return pd.DataFrame(
        {
            "itemID": [f"item_{i}" for i in range(20)],
            "text": ["machine learning for recommendations"] * 20,
        }
    )


@pytest.mark.notebooks
def test_tfidf_custom_parquet(tmp_path):
    from skill.scripts.tfidf_custom import main

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
            ]
        )
    output = f.getvalue()

    metrics = _parse_metrics(output)
    assert "status" in metrics
    assert "n_items" in metrics
    assert "n_rows" in metrics
    assert "MODEL_HANDLE=" in output


@pytest.mark.notebooks
def test_tfidf_custom_column_override(tmp_path):
    from skill.scripts.tfidf_custom import main

    df = _make_dataframe().rename(columns={"text": "description"})
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
                "--col-text",
                "description",
                "--model-out",
                "--state-root",
                str(state_root),
            ]
        )
    output = f.getvalue()

    metrics = _parse_metrics(output)
    assert "status" in metrics
    assert "MODEL_HANDLE=" in output


@pytest.mark.notebooks
def test_tfidf_custom_help_exits_zero():
    import pathlib
    import subprocess
    import sys

    script_path = pathlib.Path(__file__).parent.parent / "skill" / "scripts" / "tfidf_custom.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
