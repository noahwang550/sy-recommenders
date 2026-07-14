"""Dataset splitting MCP tools."""

import logging

from mcp_server.deps import load_splitters
from mcp_server.schemas import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL
from mcp_server.serialization import df_from_json, maybe_cache

logger = logging.getLogger("recommenders-ai")


def _split_result(train, test, cache_path):
    return {
        "train": maybe_cache(train, cache_path),
        "test": maybe_cache(test, cache_path),
    }


def register_split_tools(server):
    from mcp_server.http_transport import _TOOL_REGISTRY

    def split_random(
        data: str,
        ratio: float = 0.75,
        seed: int = 42,
        cache_path: str | None = None,
    ) -> dict:
        """Randomly split a DataFrame into train and test sets."""
        df = df_from_json(data)
        split_fn = load_splitters()["random"]
        splits = split_fn(df, ratio=ratio, seed=seed)
        if isinstance(splits, list):
            train, test = splits[0], splits[1]
        else:
            train, test = splits, None  # pragma: no cover - defensive
        logger.info(
            "split_random ratio=%s train_rows=%d test_rows=%d", ratio, len(train), len(test)
        )
        return _split_result(train, test, cache_path)

    def split_chrono(
        data: str,
        ratio: float = 0.75,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_timestamp: str = DEFAULT_TIMESTAMP_COL,
        cache_path: str | None = None,
    ) -> dict:
        """Chronologically split a DataFrame by timestamp."""
        df = df_from_json(data)
        split_fn = load_splitters()["chrono"]
        splits = split_fn(
            df,
            ratio=ratio,
            col_user=col_user,
            col_item=col_item,
            col_timestamp=col_timestamp,
        )
        if isinstance(splits, list):
            train, test = splits[0], splits[1]
        else:
            train, test = splits, None
        logger.info(
            "split_chrono ratio=%s train_rows=%d test_rows=%d", ratio, len(train), len(test)
        )
        return _split_result(train, test, cache_path)

    def split_stratified(
        data: str,
        ratio: float = 0.75,
        seed: int = 42,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        cache_path: str | None = None,
    ) -> dict:
        """Stratified split preserving per-user item distributions."""
        df = df_from_json(data)
        split_fn = load_splitters()["stratified"]
        splits = split_fn(
            df,
            ratio=ratio,
            seed=seed,
            col_user=col_user,
            col_item=col_item,
        )
        if isinstance(splits, list):
            train, test = splits[0], splits[1]
        else:
            train, test = splits, None
        logger.info(
            "split_stratified ratio=%s train_rows=%d test_rows=%d", ratio, len(train), len(test)
        )
        return _split_result(train, test, cache_path)

    def split_numpy(
        data: str,
        ratio: float = 0.75,
        seed: int = 42,
        cache_path: str | None = None,
    ) -> dict:
        """Low-level numpy stratified split (returns matrix train/test)."""
        df = df_from_json(data)
        split_fn = load_splitters()["numpy"]
        train, test = split_fn(df.values, ratio=ratio, seed=seed)
        import pandas as pd

        train_df = pd.DataFrame(train, columns=df.columns)
        test_df = pd.DataFrame(test, columns=df.columns)
        logger.info(
            "split_numpy ratio=%s train_rows=%d test_rows=%d", ratio, len(train_df), len(test_df)
        )
        return _split_result(train_df, test_df, cache_path)

    for name, fn in (
        ("split_random", split_random),
        ("split_chrono", split_chrono),
        ("split_stratified", split_stratified),
        ("split_numpy", split_numpy),
    ):
        _TOOL_REGISTRY[name] = fn
        if hasattr(server, "tool"):
            server.tool(name)(fn)
