"""Data loading MCP tools."""

import logging

from mcp_server.deps import load_criteo_loader, load_mind_api, load_movielens_loader
from mcp_server.schemas import CriteoInput, MindInput, MovielensInput
from mcp_server.serialization import maybe_cache

logger = logging.getLogger("recommenders-ai")


def register_data_tools(server):
    from mcp_server.http_transport import _TOOL_REGISTRY

    def load_movielens(size: str = "100k", cache_path: str | None = None) -> dict:
        """Load the Movielens dataset and return a serialised DataFrame payload."""
        loader = load_movielens_loader()
        df = loader(size=size)
        logger.info("load_movielens size=%s rows=%d cols=%s", size, len(df), list(df.columns))
        return maybe_cache(df, cache_path)

    def load_criteo(size: str = "sample", cache_path: str | None = None) -> dict:
        """Load the Criteo dataset and return a serialised DataFrame payload."""
        loader = load_criteo_loader()
        df = loader(size=size)
        logger.info("load_criteo size=%s rows=%d cols=%s", size, len(df), list(df.columns))
        return maybe_cache(df, cache_path)

    def load_mind(size: str = "small", dest_path: str | None = None) -> dict:
        """Download and extract the MIND dataset; return train/valid paths.

        Note: MIND is delivered as raw TSV news files, not as a single DataFrame.
        The tool returns file system paths so downstream scripts can load them.
        """
        download_mind, extract_mind = load_mind_api()
        train_zip, valid_zip = download_mind(size=size, dest_path=dest_path)
        train_path, valid_path = extract_mind(train_zip, valid_zip)
        logger.info("load_mind size=%s train=%s valid=%s", size, train_path, valid_path)
        return {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "size": size,
        }

    for name, fn in (
        ("load_movielens", load_movielens),
        ("load_criteo", load_criteo),
        ("load_mind", load_mind),
    ):
        _TOOL_REGISTRY[name] = fn
        if hasattr(server, "tool"):
            server.tool(name)(fn)

    # Expose schemas for introspection (not strictly required, but helpful).
    server._tool_schemas = getattr(server, "_tool_schemas", {})
    server._tool_schemas.update(
        {
            "load_movielens": MovielensInput,
            "load_criteo": CriteoInput,
            "load_mind": MindInput,
        }
    )
