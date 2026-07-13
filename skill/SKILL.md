# Recommenders AI Skill

Agent Skill for [Microsoft Recommenders](https://github.com/microsoft/recommenders) v1.2.1 — training scripts, playbooks, and MCP tools behind a Dockerised server.

## When to use / 何时使用

Use this skill when you want to build, evaluate, or deploy a recommendation pipeline with Microsoft Recommenders from an agent. It provides:

- **12 atomic MCP tools** for data loading, splitting, evaluation, and top-k ranking (stdio or HTTP).
- **7 runnable training scripts** derived from upstream `examples/00_quick_start/` notebooks (plus `sar_custom.py` for user-supplied data).
- **Persistent model handles** via `state.py` so trained models survive across conversations.

当需要从 agent 构建、评估或部署推荐管线时使用本 Skill。提供：

- **12 个原子 MCP 工具**（数据加载 / 划分 / 评估 / top-k 排序），stdio 或 HTTP 双传输。
- **7 个可执行训练脚本**，源自上游 `examples/00_quick_start/` 笔记本（另含 `sar_custom.py` 支持用户自有数据）。
- **持久化模型 handle**，通过 `state.py` 实现跨会话复用。

---

## Capability matrix / 能力矩阵

| Script | Source notebook | Image | GPU | MCP tools used |
|---|---|---|---|---|
| `sar_movielens.py` | `sar_movielens.ipynb` | core | no | `load_movielens`, `split_random`, `eval_ranking` |
| `ncf_movielens.py` | `ncf_movielens.ipynb` | gpu | **yes** | `load_movielens`, `split_random` |
| `sasrec_amazon.py` | `sasrec_amazon.ipynb` | gpu | **yes** | — |
| `lightgbm_tinycriteo.py` | `lightgbm_tinycriteo.ipynb` | core | no | `load_criteo` |
| `tfidf_covid.py` | `tfidf_covid.ipynb` | core | no | — |
| `eval_quickstart.py` | (various) | core | no | `load_movielens`, `split_random`, `eval_ranking` |
| `sar_custom.py` | (new — adapts sar_movielens to user data) | core | no | `load_movielens`, `split_random`, `eval_ranking` |

### MCP tools × Image tier / MCP 工具 × 镜像档

| Tool | core | gpu |
|---|:---:|:---:|
| `load_movielens` | ✅ | ✅ |
| `load_criteo` | ✅ | ✅ |
| `load_mind` | ✅ | ✅ |
| `split_random` / `split_chrono` / `split_stratified` / `split_numpy` | ✅ | ✅ |
| `eval_rating` / `eval_classification` / `eval_ranking` / `eval_beyond_accuracy` | ✅ | ✅ |
| `get_top_k` | ✅ | ✅ |
| SAR / RLRMC / TF-IDF / LightGBM / BPR (via scripts) | ✅ | ✅ |
| NCF / SASRec / Wide&Deep / deeprec / newsrec (via scripts) | ❌ → `MissingExtraError` | ✅ |
| Spark / ALS | ❌ Not supported | ❌ Not supported |

---

## 5-step playbook index / 5 步 playbook 索引

1. [`playbooks/01_prepare_data.md`](playbooks/01_prepare_data.md) — Load and cache a dataset.
2. [`playbooks/02_split.md`](playbooks/02_split.md) — Split into train/test with the right strategy.
3. [`playbooks/03_train.md`](playbooks/03_train.md) — Pick a script and run training.
4. [`playbooks/04_evaluate.md`](playbooks/04_evaluate.md) — Evaluate predictions or rankings.
5. [`playbooks/05_optimize.md`](playbooks/05_optimize.md) — Sweep parameters and persist the best model.

---

## Training artifact handles / 训练产物 handle

When a script is run with `--model-out`, it calls `state.StateStore.put_model(model)` and prints `MODEL_HANDLE=<32-hex-id>`.

**Handle storage layout** (one directory per handle under `state.root`):

```
<state.root>/<handle>/
├── meta.json       # {handle, kind, created_at, expires_at, recommends_version}
└── model.pkl       # cloudpickle serialised model object
```

- **Serialisation**: `cloudpickle` (not stdlib `pickle`) — handles lambda, closures, and cross-version compatibility better than stdlib pickle.
- **Handle id**: `secrets.token_hex(16)` — 32 hex characters, cryptographically random, not enumerable.
- **TTL**: Defaults to 24 h (`STATE_TTL_SECONDS` env). Expired handles are lazily cleaned every hour (`STATE_CLEANUP_INTERVAL` env).
- **Version check**: `state.get_model()` verifies `recommends_version` in `meta.json` matches the running server's `recommenders.__version__`. Mismatch → `StateVersionError`.
- **Concurrency**: `FileLock` per handle directory + atomic write (tmp + rename).

**Retrieving a model in a later session:**

```python
from mcp_server.state import StateStore

store = StateStore("/app/state")  # or StateStore("./state") locally
model = store.get_model("<MODEL_HANDLE>")
predictions = model.recommend_k_items(test_df, top_k=10, remove_seen=True)
```

---

## Missing-dependency handling / 缺依赖处理

When a tool or script requires an upstream extra that is not installed, the server or script raises a **structured** `MissingExtraError`:

```
MissingExtraError: Symbol 'recommenders.models.ncf.ncf_singlenode.NCF' requires extra 'gpu'.
Install with: pip install 'recommenders-ai[gpu]' or pull recommenders-mcp:gpu image.
```

### Agent resolution flow

1. **Parse** the error message for the `extra` name (e.g., `gpu`).
2. **Instruct** the user:
   - **Docker**: `docker pull recommenders-mcp:gpu` or rebuild with `--build-arg COMPUTE=gpu`.
   - **pip**: `pip install 'recommenders-ai[gpu]'`.
3. **Retry** the operation after the dependency is available.

### HTTP auth errors

HTTP mode requires `MCP_HTTP_TOKEN` environment variable. If unset, the server raises `AuthConfigError` at startup and refuses to launch (fail-closed). Requests without a valid Bearer token receive `401 {"error": "unauthorized"}`.

### Path traversal rejection

`file://` URIs pointing outside allowed roots (`cwd`, tempdir, `MCP_FILE_ROOTS`) are rejected:

```
ValueError: file:// path outside allowed roots: /etc/passwd
```

---

## Script conventions / 脚本约定

All scripts in `skill/scripts/` follow a uniform structure:

1. **Top-of-file comment** marking the source notebook: `Source: examples/00_quick_start/<name>.ipynb`.
2. **argparse** entry with common flags: `--size`, `--ratio`, `--top-k`, `--cache-path`, `--model-out`, `--state-root`.
3. **GPU scripts use lazy imports** — the `import recommenders.models.*` statement is inside `main()`, so `--help` works even in core images.
4. **JSON metrics on stdout** — parseable by the agent.
5. **Optional `--model-out`** — calls `state.put_model()` and prints `MODEL_HANDLE=<id>`.

### SAR-specific notes

- `remove_seen=True` on `recommend_k_items()` — training items are excluded from recommendations.
- Test users absent from training data are filtered before scoring (SAR cannot cold-start users).

### sar_custom.py — generic user-data entry point / 通用用户数据入口

Unlike the other scripts (which are hardcoded to specific built-in datasets), `sar_custom.py` trains SAR on a **user-supplied data file** (parquet, csv, or tsv). This is the go-to script when you want to train on your own business data.

与其他脚本（硬编码使用特定内置数据集）不同，`sar_custom.py` 在**用户自有数据文件**（parquet、csv 或 tsv）上训练 SAR。当需要在自有业务数据上训练时使用此脚本。

```bash
python skill/scripts/sar_custom.py \
  --data your_ratings.parquet \
  --col-user user_id --col-item item_id --col-rating score \
  --top-k 10 --model-out --state-root ./state
```

Required flag: `--data`. Optional column overrides: `--col-user`, `--col-item`, `--col-rating`, `--col-timestamp` (defaults match `recommenders.utils.constants`).

---

## Dependency tiers / 依赖档位

| Extra | Image | What it enables |
|---|---|---|
| *(core)* | `recommenders-mcp:core` | Data loading, splits, evaluation, SAR, TF-IDF, LightGBM, BPR |
| `[gpu]` | `recommenders-mcp:gpu` | + TensorFlow ≥ 2.8, PyTorch ≥ 2.0: NCF, SASRec, Wide&Deep, deeprec, newsrec |
| `[dev]` | *(local only)* | black, pytest, pytest-cov, pytest-mock, httpx |
| `[all]` | *(local only)* | Everything including `[experimental]` |
| `[spark]` | ❌ Not in wrapper | ALS / Spark — intentionally excluded |
