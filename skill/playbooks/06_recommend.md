# Playbook 06 — Generate Recommendations

## Goal
Score users against a persisted model and retrieve top-k recommendations.

## Steps
1. Train a model via `skill/scripts/sar_custom.py` (or `tfidf_custom.py`).
   ```bash
   python skill/scripts/sar_custom.py --data data.parquet --model-out
   ```
2. Copy the printed `MODEL_HANDLE`.
3. (Optional) Validate the handle:
   ```bash
   mcp describe_handle(handle=MODEL_HANDLE)
   ```
4. Call `recommend` with the handle and a DataFrame of users to score:
   ```bash
   mcp recommend(model_handle=MODEL_HANDLE, user_data=user_data_json, top_k=10)
   ```
5. Evaluate the recommendations with `eval_ranking`:
   ```bash
   mcp eval_ranking(rating_true=test_json, rating_pred=recs_json, k=10)
   ```

## Stateless-Atoms Boundary
- The wrapper does **not** write to the agent's own database.
- `recommend` returns `uri`, `rows`, `schema`, `skipped_user_count`, and `model_handle`.
- Large result sets can be returned via `cache_path` as a `file://` parquet URI.
- The agent is responsible for persisting results to its own store.
