# Playbook 01 — Prepare Data

## Goal
Load a dataset into a DataFrame and optionally cache it on disk.

## Steps
1. Choose a dataset:
   - Movielens: `load_movielens(size="100k")`
   - Criteo: `load_criteo(size="sample")`
   - MIND: `load_mind(size="small")`
2. If the DataFrame is larger than 50,000 rows, provide a `cache_path` to trigger
   parquet-backed transport.
3. Verify the returned payload contains `uri`, `rows`, and `schema`.

## Example
```json
{
  "tool": "load_movielens",
  "arguments": {
    "size": "100k",
    "cache_path": "/tmp/recommenders"
  }
}
```

## Output
```json
{
  "uri": "file:///tmp/recommenders/a1b2c3d4.parquet",
  "rows": 100000,
  "schema": {"userID": "int64", "itemID": "int64", "rating": "float64"}
}
```
