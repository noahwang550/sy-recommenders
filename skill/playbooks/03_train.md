# Playbook 03 — Train a Model

## Goal
Run one of the provided training scripts to fit a model and obtain a handle.

## Steps
1. Choose a script from `skill/scripts/` matching your dataset and compute budget.
2. For CPU: `sar_movielens.py`, `sar_custom.py` (user data), `lightgbm_tinycriteo.py`, `tfidf_covid.py`.
3. For GPU: `ncf_movielens.py`, `sasrec_amazon.py`.
4. Run with `--model-out` to persist the trained model.

## Example
```bash
python skill/scripts/sar_movielens.py --size 100k --model-out
```

## Output
```json
{
  "precision": 0.330753,
  "recall": 0.176385,
  "ndcg": 0.382461,
  "map": 0.110591
}
MODEL_HANDLE=aabbccddeeff00112233445566778899
```

Save the `MODEL_HANDLE` for later retrieval.
