"""GPU smoke tests for SASRec on Amazon reviews (nightly)."""

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.mark.gpu
def test_smoke_sasrec_amazon():
    pytest.skip("GPU smoke test: run manually with pytest -m gpu")
