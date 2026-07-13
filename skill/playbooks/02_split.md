# Playbook 02 — Split Data

## Goal
Split a loaded DataFrame into train and test sets.

## Steps
1. Use the `uri` from the data-loading step as the `data` argument.
2. Choose a splitter:
   - `split_random`: random holdout
   - `split_chrono`: chronological split by timestamp
   - `split_stratified`: stratified per user
   - `split_numpy`: numpy stratified variant
3. Set `ratio` (default 0.75 for train).

## Example
```json
{
  "tool": "split_random",
  "arguments": {
    "data": "file:///tmp/recommenders/a1b2c3d4.parquet",
    "ratio": 0.75,
    "seed": 42
  }
}
```

## Output
```json
{
  "train": {"uri": "...", "rows": 75000, "schema": {}},
  "test": {"uri": "...", "rows": 25000, "schema": {}}
}
```
