# Playbook 04 — Evaluate

## Goal
Evaluate predicted ratings or rankings against ground truth.

## Steps
1. For rating metrics, use `eval_rating`.
2. For classification metrics, use `eval_classification`.
3. For ranking metrics (precision, recall, nDCG, MAP), use `eval_ranking`.
4. For beyond-accuracy metrics, use `eval_beyond_accuracy`.

## Example
```json
{
  "tool": "eval_ranking",
  "arguments": {
    "rating_true": "<test_uri>",
    "rating_pred": "<pred_uri>",
    "k": 10
  }
}
```

## Output
```json
{
  "precision": 0.33,
  "recall": 0.18,
  "ndcg": 0.38,
  "map": 0.11,
  "r_precision": 0.25
}
```

Note: `eval_ranking` uses `map` (not `map_at_k`) to align with the upstream
SAR Movielens 100k baseline.
