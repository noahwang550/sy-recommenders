"""Persistent state storage for DataFrames and trained models.

A ``StateStore`` writes each artifact as a handle directory containing:
  - meta.json: metadata, including ``recommends_version``
  - data.parquet for DataFrames, or model.pkl for model objects

Handle ids are generated with ``secrets.token_hex(16)`` (32 hex chars).
Model pickles are only ever written by this module's ``put_model`` and
read by ``get_model``; they are never accepted as MCP tool input.
"""

import json
import os
import secrets
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cloudpickle
from filelock import FileLock

DEFAULT_TTL = int(os.environ.get("STATE_TTL_SECONDS", "86400"))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("STATE_CLEANUP_INTERVAL", "3600"))
_last_cleanup = 0.0
_last_cleanup_lock = threading.Lock()


def _recommenders_version() -> str:
    try:
        import recommenders

        return recommenders.__version__
    except Exception:  # pragma: no cover
        return "unknown"


class StateNotFoundError(FileNotFoundError):
    """Raised when a requested handle does not exist or has expired."""


class StateVersionError(ValueError):
    """Raised when a model checkpoint was written with a different recommenders version."""

    def __init__(self, expected: str, found: str):
        self.expected = expected
        self.found = found
        super().__init__(
            f"Model checkpoint expects recommenders {found}, but server expects {expected}"
        )


class StateStore:
    def __init__(self, root: str, ttl_seconds: int = DEFAULT_TTL):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _handle_dir(self, handle: str) -> Path:
        if len(handle) != 32 or not all(c in "0123456789abcdef" for c in handle.lower()):
            raise ValueError("Invalid handle id")
        return self.root / handle

    def _new_handle(self) -> str:
        while True:
            handle = secrets.token_hex(16)
            if not self._handle_dir(handle).exists():
                return handle

    def _write_meta(self, handle_dir: Path, kind: str, extra: dict | None = None) -> None:
        now = datetime.now(timezone.utc)
        meta = {
            "handle": handle_dir.name,
            "kind": kind,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.ttl)).isoformat(),
            "recommends_version": _recommenders_version(),
        }
        if extra:
            meta.update(extra)
        tmp = handle_dir / "meta.json.tmp"
        final = handle_dir / "meta.json"
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(final)

    def put_df(self, df, kind_label: str = "df") -> str:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("put_df expects a pandas DataFrame")

        self._maybe_cleanup()
        handle = self._new_handle()
        handle_dir = self._handle_dir(handle)
        handle_dir.mkdir(parents=True)
        lock = FileLock(str(handle_dir / ".lock"))
        with lock:
            df.to_parquet(handle_dir / "data.parquet")
            self._write_meta(handle_dir, "df")
        return handle

    def get_df(self, handle: str):
        import pandas as pd

        handle_dir = self._handle_dir(handle)
        if not handle_dir.exists():
            raise StateNotFoundError(handle)
        lock = FileLock(str(handle_dir / ".lock"))
        with lock:
            return pd.read_parquet(handle_dir / "data.parquet")

    def put_model(self, model, recommends_version: str | None = None) -> str:
        if recommends_version is None:
            recommends_version = _recommenders_version()

        self._maybe_cleanup()
        handle = self._new_handle()
        handle_dir = self._handle_dir(handle)
        handle_dir.mkdir(parents=True)
        lock = FileLock(str(handle_dir / ".lock"))
        with lock:
            tmp = handle_dir / "model.pkl.tmp"
            with open(tmp, "wb") as f:
                cloudpickle.dump(model, f)
            tmp.replace(handle_dir / "model.pkl")
            self._write_meta(handle_dir, "model")
        return handle

    def get_model(self, handle: str, expects_version: str | None = None):
        if expects_version is None:
            expects_version = _recommenders_version()

        handle_dir = self._handle_dir(handle)
        if not handle_dir.exists():
            raise StateNotFoundError(handle)
        lock = FileLock(str(handle_dir / ".lock"))
        with lock:
            meta = json.loads((handle_dir / "meta.json").read_text())
            stored_version = meta.get("recommends_version")
            if stored_version != expects_version:
                raise StateVersionError(expected=expects_version, found=stored_version)
            with open(handle_dir / "model.pkl", "rb") as f:
                return cloudpickle.load(f)

    def _maybe_cleanup(self) -> None:
        global _last_cleanup
        with _last_cleanup_lock:
            now = time.time()
            if now - _last_cleanup < CLEANUP_INTERVAL_SECONDS:
                return
            _last_cleanup = now
        self.cleanup_expired()

    def cleanup_expired(self) -> list[str]:
        removed = []
        now = datetime.now(timezone.utc)
        for handle_dir in self.root.iterdir():
            if not handle_dir.is_dir():
                continue
            meta_path = handle_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                expires_at = datetime.fromisoformat(meta["expires_at"])
                if expires_at <= now:
                    shutil.rmtree(handle_dir)
                    removed.append(handle_dir.name)
            except (KeyError, ValueError, OSError):
                continue
        return removed

    def exists(self, handle: str) -> bool:
        try:
            return self._handle_dir(handle).exists()
        except ValueError:
            return False

    def list_handles(self, kind: str | None = None) -> list[dict]:
        self._maybe_cleanup()
        result: list[dict] = []
        now = datetime.now(timezone.utc)
        for handle_dir in self.root.iterdir():
            if not handle_dir.is_dir():
                continue
            try:
                meta_path = handle_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                expires_at = datetime.fromisoformat(meta["expires_at"])
                if expires_at <= now:
                    continue
                if kind is not None and meta.get("kind") != kind:
                    continue
                result.append(
                    {
                        "handle": meta["handle"],
                        "kind": meta["kind"],
                        "created_at": meta["created_at"],
                        "expires_at": meta["expires_at"],
                        "recommends_version": meta["recommends_version"],
                    }
                )
            except (OSError, KeyError, ValueError):
                continue
        return result

    def describe_handle(self, handle: str) -> dict:
        handle_dir = self._handle_dir(handle)
        if not handle_dir.exists():
            raise StateNotFoundError(handle)
        meta = json.loads((handle_dir / "meta.json").read_text())
        if (handle_dir / "data.parquet").exists():
            size_bytes = (handle_dir / "data.parquet").stat().st_size
        elif (handle_dir / "model.pkl").exists():
            size_bytes = (handle_dir / "model.pkl").stat().st_size
        else:
            size_bytes = 0
        return {
            "handle": handle,
            "kind": meta["kind"],
            "created_at": meta["created_at"],
            "expires_at": meta["expires_at"],
            "recommends_version": meta["recommends_version"],
            "size_bytes": size_bytes,
        }

    def delete_handle(self, handle: str) -> bool:
        handle_dir = self._handle_dir(handle)
        if not handle_dir.exists():
            return False
        shutil.rmtree(handle_dir)
        return True
