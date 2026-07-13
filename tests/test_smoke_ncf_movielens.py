"""GPU smoke tests (nightly)."""

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.mark.gpu
def test_smoke_ncf_movielens():
    pytest.skip("GPU smoke test: run manually with pytest -m gpu")
