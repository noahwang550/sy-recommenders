"""Typed request/response schemas used by MCP tool wrappers.

These Pydantic models document tool signatures and give clients a stable
contract.  The server itself serialises them to plain dicts before
registering tools with the MCP SDK.  Default column names mirror
recommenders.utils.constants but are duplicated here so this module does
not need to import the (heavy) upstream package at import time.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_K = 10


class DataFramePayload(BaseModel):
    """A DataFrame transported as JSON (orient=split) or a file:// URI."""

    # `populate_by_name` lets callers construct with either the python
    # attribute (`schema_info=...`) or the serialised contract key
    # (`schema=...`).  The field attribute is intentionally not named
    # ``schema`` to avoid shadowing a member on pydantic v2's BaseModel
    # (which raises a UserWarning on import); the ``serialization_alias``
    # keeps the on-the-wire key as ``schema`` so the tool output contract
    # is unchanged.
    model_config = ConfigDict(populate_by_name=True)

    data: str = Field(
        ...,
        description="JSON string (orient=split) or file:// URI pointing to a parquet/json file.",
    )
    rows: Optional[int] = Field(None, description="Number of rows, provided as a hint.")
    schema_info: Optional[dict] = Field(
        None,
        serialization_alias="schema",
        description="Schema/dtype map, provided as a hint. Serialises to the `schema` key.",
    )


class MovielensInput(BaseModel):
    size: str = Field(default="100k", description="Movielens size: mock100, 100k, 1m, 10m, 20m.")
    cache_path: Optional[str] = Field(None, description="Optional directory to cache the raw df.")


class CriteoInput(BaseModel):
    size: str = Field(default="sample", description="Criteo size: sample or full.")
    cache_path: Optional[str] = Field(None, description="Optional directory to cache the raw df.")


class MindInput(BaseModel):
    size: str = Field(default="small", description="MIND size: small, large, demo.")
    dest_path: Optional[str] = Field(None, description="Directory to download MIND data into.")


class SplitInput(BaseModel):
    data: str = Field(..., description="DataFrame JSON or file:// URI.")
    ratio: float = Field(default=0.75, description="Fraction of data for the train split.")
    seed: int = Field(default=42, description="Random seed where applicable.")
    col_user: str = Field(default=DEFAULT_USER_COL)
    col_item: str = Field(default=DEFAULT_ITEM_COL)
    col_timestamp: str = Field(default=DEFAULT_TIMESTAMP_COL)


class EvalRatingInput(BaseModel):
    rating_true: str = Field(..., description="Ground-truth DataFrame JSON or file:// URI.")
    rating_pred: str = Field(..., description="Predicted ratings DataFrame JSON or file:// URI.")
    col_user: str = Field(default=DEFAULT_USER_COL)
    col_item: str = Field(default=DEFAULT_ITEM_COL)
    col_rating: str = Field(default=DEFAULT_RATING_COL)
    col_prediction: str = Field(default=DEFAULT_PREDICTION_COL)


class EvalRankingInput(BaseModel):
    rating_true: str = Field(..., description="Ground-truth DataFrame JSON or file:// URI.")
    rating_pred: str = Field(..., description="Predicted ranking DataFrame JSON or file:// URI.")
    col_user: str = Field(default=DEFAULT_USER_COL)
    col_item: str = Field(default=DEFAULT_ITEM_COL)
    col_prediction: str = Field(default=DEFAULT_PREDICTION_COL)
    k: int = Field(default=DEFAULT_K, description="Top-k cutoff.")


class EvalBeyondAccuracyInput(BaseModel):
    train_df: str = Field(..., description="Training DataFrame JSON or file:// URI.")
    reco_df: str = Field(..., description="Recommended top-k DataFrame JSON or file:// URI.")
    col_user: str = Field(default=DEFAULT_USER_COL)
    col_item: str = Field(default=DEFAULT_ITEM_COL)


class GetTopKInput(BaseModel):
    data: str = Field(..., description="DataFrame JSON or file:// URI.")
    col_user: str = Field(default=DEFAULT_USER_COL)
    col_rating: str = Field(default=DEFAULT_RATING_COL)
    k: int = Field(default=DEFAULT_K)


class SplitOutput(BaseModel):
    train: str = Field(..., description="Train DataFrame JSON or file:// URI.")
    test: str = Field(..., description="Test DataFrame JSON or file:// URI.")


class RankingMetricsOutput(BaseModel):
    precision: float
    recall: float
    ndcg: float
    map: float
    r_precision: float


class RatingMetricsOutput(BaseModel):
    rmse: float
    mae: float
    rsquared: float
    exp_var: float


class ClassificationMetricsOutput(BaseModel):
    auc: float
    logloss: float


class BeyondAccuracyOutput(BaseModel):
    diversity: float
    novelty: float
    serendipity: float
    catalog_coverage: float
    distributional_coverage: float
