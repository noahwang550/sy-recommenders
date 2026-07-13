# Playbook 05 — Optimize

## Goal
Sweep parameters and persist the best model.

## Steps
1. Pick a parameter grid (e.g., `similarity_type` for SAR).
2. Loop over the grid inside a shell script or notebook cell.
3. Run `eval_quickstart.py` or the relevant training script for each setting.
4. Compare metrics and keep the best run.
5. Persist the best model with `--model-out` and record its handle.

## Example
```bash
for sim in jaccard cosine; do
  python skill/scripts/sar_movielens.py --size 100k --similarity-type "$sim"
done
```

## Output
The run with the highest `map` is the winner; use its `MODEL_HANDLE` for
deployment or further analysis.
