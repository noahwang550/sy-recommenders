"""Tests for skill.scripts.tfidf_covid.

These tests avoid network access by mocking the COVID-19 data loader and the
TfidfRecommender class. They verify that the script constructs the recommender
with the required ``id_col`` argument, which is the fix for the no-arg bug.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("recommenders") is None,
    reason="recommenders not installed; skill scripts require upstream package",
)


def test_tfidf_covid_passes_id_col_to_recommender():
    """TfidfRecommender must receive id_col; previously it was called with no args."""
    import pandas as pd

    from skill.scripts import tfidf_covid

    df = pd.DataFrame({"itemID": ["a", "b"], "text": ["foo", "bar"]})

    mock_recommender_cls = MagicMock()
    mock_recommender = mock_recommender_cls.return_value
    mock_recommender.recommend_top_k_items.return_value = pd.DataFrame({"x": [1]})

    with (
        patch.object(tfidf_covid, "load_covid19_data", return_value=df),
        patch.object(tfidf_covid, "TfidfRecommender", mock_recommender_cls),
    ):
        result = tfidf_covid.main([])

    assert result == 0
    mock_recommender_cls.assert_called_once_with(id_col="itemID")
    mock_recommender.fit.assert_called_once_with(df["text"], df["itemID"])
    mock_recommender.recommend_top_k_items.assert_called_once_with(df, k=5)


def test_tfidf_covid_uses_custom_col_item():
    """The --col-item argument should be passed through to TfidfRecommender."""
    import pandas as pd

    from skill.scripts import tfidf_covid

    df = pd.DataFrame({"doc_id": ["a", "b"], "text": ["foo", "bar"]})

    mock_recommender_cls = MagicMock()
    mock_recommender = mock_recommender_cls.return_value
    mock_recommender.recommend_top_k_items.return_value = pd.DataFrame({"x": [1]})

    with (
        patch.object(tfidf_covid, "load_covid19_data", return_value=df),
        patch.object(tfidf_covid, "TfidfRecommender", mock_recommender_cls),
    ):
        result = tfidf_covid.main(["--col-item", "doc_id"])

    assert result == 0
    mock_recommender_cls.assert_called_once_with(id_col="doc_id")
    mock_recommender.fit.assert_called_once_with(df["text"], df["doc_id"])
