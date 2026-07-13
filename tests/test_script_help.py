"""Tests that all skill scripts have a working --help."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

SCRIPT_DIR = pathlib.Path(__file__).parent.parent / "skill" / "scripts"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("recommenders") is None,
    reason="recommenders not installed; skill scripts require upstream package",
)


@pytest.mark.parametrize("script", [
    "sar_movielens.py",
    "sar_custom.py",
    "ncf_movielens.py",
    "sasrec_amazon.py",
    "lightgbm_tinycriteo.py",
    "tfidf_covid.py",
    "eval_quickstart.py",
])
def test_script_help_exits_zero(script):
    path = SCRIPT_DIR / script
    result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
