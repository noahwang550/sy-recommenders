"""Smoke test aligning with upstream SAR Movielens 100k baseline."""

import importlib.util
import json
import os
import re
import sys

import pytest


class _NoInternet:
    def __init__(self):
        self._ok = None

    def __bool__(self):
        if self._ok is None:
            import urllib.request

            try:
                urllib.request.urlopen("https://files.grouplens.org", timeout=3)
                self._ok = True
            except Exception:
                self._ok = False
        return self._ok


_HAS_INTERNET = _NoInternet()

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


@pytest.mark.notebooks
@pytest.mark.skipif(not _HAS_INTERNET, reason="No internet access for Movielens download")
def test_sar_movielens_100k_baseline():
    import io
    import contextlib
    from skill.scripts.sar_movielens import main

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        main(["--size", "100k"])
    output = f.getvalue()
    metrics = _parse_metrics(output)
    assert metrics["precision"] == pytest.approx(0.330753, rel=0.05, abs=0.05)
    assert metrics["recall"] == pytest.approx(0.176385, rel=0.05, abs=0.05)
    assert metrics["ndcg"] == pytest.approx(0.382461, rel=0.05, abs=0.05)
    assert metrics["map"] == pytest.approx(0.110591, rel=0.05, abs=0.05)


def test_sar_movielens_mock100_synthetic():
    import io
    import contextlib
    from skill.scripts.sar_movielens import main

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        main(["--size", "mock100"])
    output = f.getvalue()
    metrics = _parse_metrics(output)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "ndcg" in metrics
    assert "map" in metrics
