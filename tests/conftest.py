"""Shared pytest fixtures for the recommenders-ai wrapper."""

import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def temp_state_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "userID": [1, 1, 2, 2, 3],
            "itemID": [10, 11, 10, 12, 11],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.0],
        }
    )


@pytest.fixture
def sample_df_with_ts():
    return pd.DataFrame(
        {
            "userID": [1, 1, 2, 2, 3],
            "itemID": [10, 11, 10, 12, 11],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.0],
            "timestamp": [1, 2, 1, 2, 3],
        }
    )


@pytest.fixture
def rating_true_pred():
    true = pd.DataFrame(
        {
            "userID": [1, 1, 2, 2],
            "itemID": [10, 11, 10, 12],
            "rating": [4.0, 3.0, 5.0, 2.0],
        }
    )
    pred = pd.DataFrame(
        {
            "userID": [1, 1, 2, 2],
            "itemID": [10, 11, 10, 12],
            "prediction": [3.9, 3.1, 4.8, 2.2],
        }
    )
    return true, pred
