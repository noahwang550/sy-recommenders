# Tools Reference / 工具参考

The MCP server exposes **12 atomic tools** grouped into four domains. All tools are stateless pure functions — training state lives in `state.py` handles, not in the server process.

MCP server 暴露 **12 个原子工具**，分四个领域。所有工具均为无状态纯函数 — 训练状态由 `state.py` handle 管理，不驻留在 server 进程。

---

## Table of contents

- [Common concepts / 通用概念](#common-concepts)
- [Data tools (3) / 数据工具](#data-tools-3)
- [Split tools (4) / 划分工具](#split-tools-4)
- [Evaluate tools (4) / 评估工具](#evaluate-tools-4)
- [Ranking tools (1) / 排序工具](#ranking-tools-1)
- [Error codes / 错误码](#error-codes)

---

## Common concepts

### DataFrame payload

Every tool that accepts or returns a DataFrame uses a **payload dict**:

```json
{
  "uri": "<JSON string or file:// URI>",
  "rows": 100000,
  "schema": {"userID": "int64", "itemID": "int64", "rating": "float64"}
}
```

- **Small DataFrames** (≤ 50 000 rows): `uri` is a JSON string (`orient="split"`).
- **Large DataFrames** (> 50 000 rows, only when `cache_path` is provided): `uri` is a `file://` URI pointing to a parquet file on disk.

### `file://` mode and path traversal protection

The `file://` URI scheme lets tools read/write DataFrames from disk, avoiding MB-scale JSON round-trips. Paths are validated against:

1. **Process-generated files** — files written by `maybe_cache` in the current process are always allowed.
2. **Allowed roots** — the current working directory, the system temp directory, plus any paths listed in the `MCP_FILE_ROOTS` environment variable (OS-separated).
3. Paths outside these roots are **rejected** with `ValueError: file:// path outside allowed roots`.

All `file://` paths must be **absolute**. Relative paths are rejected.

### Column-name contract

Default column names mirror `recommenders.utils.constants`:

| Constant | Default value |
|---|---|
| `DEFAULT_USER_COL` | `"userID"` |
| `DEFAULT_ITEM_COL` | `"itemID"` |
| `DEFAULT_RATING_COL` | `"rating"` |
| `DEFAULT_PREDICTION_COL` | `"prediction"` |
| `DEFAULT_TIMESTAMP_COL` | `"timestamp"` |
| `DEFAULT_K` | `10` |

Tools accept `col_user`, `col_item`, etc. as keyword arguments to override these defaults.

---

## Data tools (3)

### `load_movielens`

Load the Movielens dataset and return a serialised DataFrame payload.

**Signature:**

| Parameter | Type | Default | Values | Description |
|---|---|---|---|---|
| `size` | `str` | `"100k"` | `mock100`, `100k`, `1m`, `10m`, `20m` | Dataset size |
| `cache_path` | `str \| None` | `None` | any writable directory | Directory for parquet caching (used only when rows > 50 000) |

**Returns:** `{"uri": str, "rows": int, "schema": dict}`

**Example:**

```json
// Request
{"tool": "load_movielens", "arguments": {"size": "100k", "cache_path": "/tmp/recommenders"}}

// Response (large df → file:// mode)
{"uri": "file:///tmp/recommenders/a1b2c3d4e5f6.parquet", "rows": 100000, "schema": {"userID": "int64", "itemID": "int64", "rating": "float64", "timestamp": "int64"}}
```

---

### `load_criteo`

Load the Criteo dataset and return a serialised DataFrame payload.

**Signature:**

| Parameter | Type | Default | Values | Description |
|---|---|---|---|---|
| `size` | `str` | `"sample"` | `sample`, `full` | Dataset size |
| `cache_path` | `str \| None` | `None` | any writable directory | Directory for parquet caching |

**Returns:** `{"uri": str, "rows": int, "schema": dict}`

**Example:**

```json
// Request
{"tool": "load_criteo", "arguments": {"size": "sample"}}

// Response
{"uri": "{\"columns\":[...],...}", "rows": 1000, "schema": {"label": "int64", "...": "..."}}
```

---

### `load_mind`

Download and extract the MIND (Microsoft News Dataset). Unlike Movielens/Criteo, MIND is delivered as raw TSV news files, not as a single DataFrame. The tool returns file system paths so downstream scripts can load them.

**Signature:**

| Parameter | Type | Default | Values | Description |
|---|---|---|---|---|
| `size` | `str` | `"small"` | `small`, `large`, `demo` | Dataset size |
| `dest_path` | `str \| None` | `None` | any writable directory | Directory to download into |

**Returns:** `{"train_path": str, "valid_path": str, "size": str}`

**Example:**

```json
// Request
{"tool": "load_mind", "arguments": {"size": "demo", "dest_path": "/tmp/mind"}}

// Response
{"train_path": "/tmp/mind/train", "valid_path": "/tmp/mind/valid", "size": "demo"}
```

---

## Split tools (4)

All split tools accept a DataFrame payload string (JSON or `file://` URI) as `data` and return `{"train": payload, "test": payload}`.

### `split_random`

Randomly split a DataFrame into train and test sets.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` | *(required)* | DataFrame JSON or `file://` URI |
| `ratio` | `float` | `0.75` | Train fraction |
| `seed` | `int` | `42` | Random seed |
| `cache_path` | `str \| None` | `None` | Directory for parquet caching of output |

**Returns:** `{"train": {"uri": ..., "rows": ..., "schema": ...}, "test": {"uri": ..., "rows": ..., "schema": ...}}`

**Note:** Handles `python_random_split` returning either a `list` or a single `DataFrame` — the tool always unpacks `splits[0]` / `splits[1]`.

**Example:**

```json
// Request
{"tool": "split_random", "arguments": {"data": "file:///tmp/recommenders/a1b2.parquet", "ratio": 0.8, "seed": 42}}

// Response
{
  "train": {"uri": "...", "rows": 80000, "schema": {}},
  "test":  {"uri": "...", "rows": 20000, "schema": {}}
}
```

---

### `split_chrono`

Chronological split by timestamp — earlier interactions go to train, later to test.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` | *(required)* | DataFrame JSON or `file://` URI |
| `ratio` | `float` | `0.75` | Train fraction |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |
| `col_timestamp` | `str` | `"timestamp"` | Timestamp column |
| `cache_path` | `str \| None` | `None` | Directory for output caching |

**Returns:** `{"train": payload, "test": payload}`

**Note:** This splitter is **deterministic and non-random** — the `seed` parameter is not applicable.

---

### `split_stratified`

Stratified split preserving per-user item distributions.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` | *(required)* | DataFrame JSON or `file://` URI |
| `ratio` | `float` | `0.75` | Train fraction |
| `seed` | `int` | `42` | Random seed |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |
| `cache_path` | `str \| None` | `None` | Directory for output caching |

**Returns:** `{"train": payload, "test": payload}`

---

### `split_numpy`

Low-level numpy stratified split. Operates on the raw `.values` matrix and re-wraps the result into DataFrames with the original column names.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` | *(required)* | DataFrame JSON or `file://` URI |
| `ratio` | `float` | `0.75` | Train fraction |
| `seed` | `int` | `42` | Random seed |
| `cache_path` | `str \| None` | `None` | Directory for output caching |

**Returns:** `{"train": payload, "test": payload}`

---

## Evaluate tools (4)

### `eval_rating`

Compute rating-prediction metrics: RMSE, MAE, R-squared, and explained variance.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rating_true` | `str` | *(required)* | Ground-truth DataFrame JSON or `file://` URI |
| `rating_pred` | `str` | *(required)* | Predicted ratings DataFrame JSON or `file://` URI |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |
| `col_rating` | `str` | `"rating"` | True rating column |
| `col_prediction` | `str` | `"prediction"` | Predicted rating column |

**Returns:** `{"rmse": float, "mae": float, "rsquared": float, "exp_var": float}`

**Example:**

```json
// Request
{"tool": "eval_rating", "arguments": {"rating_true": "<true_json>", "rating_pred": "<pred_json>"}}

// Response
{"rmse": 0.942, "mae": 0.738, "rsquared": 0.285, "exp_var": 0.291}
```

---

### `eval_classification`

Compute classification metrics: AUC and logloss.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rating_true` | `str` | *(required)* | Ground-truth DataFrame JSON or `file://` URI |
| `rating_pred` | `str` | *(required)* | Predicted scores DataFrame JSON or `file://` URI |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |
| `col_rating` | `str` | `"rating"` | True rating column |
| `col_prediction` | `str` | `"prediction"` | Predicted score column |

**Returns:** `{"auc": float, "logloss": float}`

---

### `eval_ranking`

Compute ranking metrics: precision, recall, nDCG, MAP, and r-precision at k.

**Important:** This tool uses `map` (not `map_at_k`) to align with the SAR Movielens 100k baseline value of ≈ 0.110591.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rating_true` | `str` | *(required)* | Ground-truth DataFrame JSON or `file://` URI |
| `rating_pred` | `str` | *(required)* | Predicted ranking DataFrame JSON or `file://` URI |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |
| `col_prediction` | `str` | `"prediction"` | Prediction score column |
| `k` | `int` | `10` | Top-k cutoff |

**Note:** Unlike `eval_rating` and `eval_classification`, this tool does **not** accept `col_rating` — the upstream ranking functions only need `col_prediction`.

**Returns:** `{"precision": float, "recall": float, "ndcg": float, "map": float, "r_precision": float}`

**Example:**

```json
// Request
{"tool": "eval_ranking", "arguments": {"rating_true": "<test_uri>", "rating_pred": "<pred_uri>", "k": 10}}

// Response
{"precision": 0.330753, "recall": 0.176385, "ndcg": 0.382461, "map": 0.110591, "r_precision": 0.247}
```

---

### `eval_beyond_accuracy`

Compute beyond-accuracy metrics: diversity, novelty, serendipity, and catalog coverage.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_df` | `str` | *(required)* | Training DataFrame JSON or `file://` URI |
| `reco_df` | `str` | *(required)* | Recommended top-k DataFrame JSON or `file://` URI |
| `col_user` | `str` | `"userID"` | User column |
| `col_item` | `str` | `"itemID"` | Item column |

**Returns:** `{"diversity": float, "novelty": float, "serendipity": float, "catalog_coverage": float, "distributional_coverage": float}`

---

## Ranking tools (1)

### `get_top_k`

Return the top-k items per user, adding a `rank` column.

**Signature:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` | *(required)* | DataFrame JSON or `file://` URI |
| `col_user` | `str` | `"userID"` | User column |
| `col_rating` | `str` | `"rating"` | Rating/score column to sort by |
| `k` | `int` | `10` | Number of top items per user |
| `cache_path` | `str \| None` | `None` | Directory for output caching |

**Returns:** `{"uri": str, "rows": int, "schema": dict}` — the DataFrame contains an additional `rank` column.

**Example:**

```json
// Request
{"tool": "get_top_k", "arguments": {"data": "<pred_uri>", "k": 5}}

// Response
{"uri": "...", "rows": 5000, "schema": {"userID": "int64", "itemID": "int64", "prediction": "float64", "rank": "int64"}}
```

---

## Error codes

Tools raise structured errors that MCP clients can parse:

### `MissingExtraError`

Raised when a tool requires an upstream extra that is not installed (e.g., GPU model in a core image).

```json
{
  "error": "Symbol 'recommenders.models.ncf.ncf_singlenode.NCF' requires extra 'gpu'. Install with: pip install 'recommenders-ai[gpu]' or pull recommenders-mcp:gpu image."
}
```

**Fields:** `extra` (str), `symbol` (str), `hint` (str).

**Resolution:** Install the matching extra or pull the correct Docker image tier.

### `StateNotFoundError`

Raised when a `state.py` handle does not exist or has expired (TTL elapsed).

```json
{
  "error": "StateNotFoundError: aabbccddeeff00112233445566778899"
}
```

**Resolution:** Re-run the training script with `--model-out` to produce a fresh handle.

### `StateVersionError`

Raised when a model checkpoint was written with a different `recommenders` version than the running server expects.

```json
{
  "error": "Model checkpoint expects recommenders 1.2.1, but server expects 2.0.0"
}
```

**Fields:** `expected` (str), `found` (str).

**Resolution:** Use a matching image version, or re-train the model.

### `AuthConfigError`

Raised at server startup when `MCP_TRANSPORT=http` but `MCP_HTTP_TOKEN` is unset or empty. The server refuses to start (fail-closed).

```
AuthConfigError: HTTP transport requires MCP_HTTP_TOKEN environment variable. Set it before starting the server.
```

### Path traversal rejection

`file://` URIs pointing outside allowed roots are rejected:

```
ValueError: file:// path outside allowed roots: /etc/passwd
```

Non-absolute paths are also rejected:

```
ValueError: file:// URI must use an absolute path: relative/path.parquet
```

### HTTP transport errors

| Status code | Condition | Response body |
|---|---|---|
| `401` | Missing or invalid Bearer token | `{"error": "unauthorized"}` |
| `404` | Unknown tool name | `{"error": "tool not found"}` |
| `500` | Tool raised an exception | `{"error": "<exception message>"}` |
