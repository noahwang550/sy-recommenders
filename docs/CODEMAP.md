# CODEMAP — recommenders-ai

Repository layout and module dependency graph. Updated to match the current codebase.

仓库目录结构与模块依赖关系图。与当前代码库同步。

---

## Directory tree / 目录树

```
recommenders-ai/
├── pyproject.toml                    # Package metadata, deps, console script, pytest config
├── Dockerfile                        # Multi-stage build: base → core | gpu (COMPUTE build arg)
├── docker-compose.yml                # Two services: stdio + http, shared mcp_state volume
├── .mcp.json                         # MCP client config sample (stdio + HTTP Bearer)
├── README.md                         # Bilingual project docs (EN + ZH)
│
├── mcp_server/                       # MCP server package
│   ├── __init__.py                   # Version metadata only (__version__ = "0.1.0")
│   ├── server.py                     # Entry point: _register_all(), main() → stdio or http
│   ├── deps.py                       # Lazy import boundary (ONLY place that imports recommenders.*)
│   ├── schemas.py                    # Pydantic models for tool I/O contracts
│   ├── serialization.py              # DataFrame JSON/parquet + file:// + path traversal guard
│   ├── state.py                      # Persistent handles: cloudpickle (models) + parquet (dfs) + TTL
│   ├── auth.py                       # HTTP Bearer token auth (secrets.compare_digest)
│   ├── http_transport.py             # FastAPI ASGI app: /health, /tools, /invoke + auth middleware
│   └── tools/                        # 12 atomic MCP tools
│       ├── __init__.py               # Re-exports register_*_tools functions
│       ├── data.py                   # load_movielens, load_criteo, load_mind (3 tools)
│       ├── split.py                  # split_random, split_chrono, split_stratified, split_numpy (4 tools)
│       ├── evaluate.py               # eval_rating, eval_classification, eval_ranking, eval_beyond_accuracy (4 tools)
│       └── ranking.py                # get_top_k (1 tool)
│
├── skill/                            # Agent Skill package
│   ├── SKILL.md                      # Capability matrix, playbook index, artifact semantics
│   ├── playbooks/                    # 5-step workflow docs
│   │   ├── 01_prepare_data.md
│   │   ├── 02_split.md
│   │   ├── 03_train.md
│   │   ├── 04_evaluate.md
│   │   └── 05_optimize.md
│   ├── scripts/                      # 6 runnable training scripts (source-tagged)
│   │   ├── sar_movielens.py          # core — SARSingleNode
│   │   ├── ncf_movielens.py          # gpu  — NCF (lazy import)
│   │   ├── sasrec_amazon.py          # gpu  — SASREC (lazy import)
│   │   ├── lightgbm_tinycriteo.py    # core — LightGBM + NumEncoder
│   │   ├── tfidf_covid.py            # core — TfidfRecommender
│   │   └── eval_quickstart.py        # core — SAR eval baseline (mock100 default)
│   └── snippets/                     # Short code fragments
│       ├── load_movielens.md
│       ├── split_random.md
│       └── eval_ranking.md
│
├── tests/                            # pytest test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures: temp_state_root, sample_df, rating_true_pred
│   ├── test_groups.yml               # CI group assignments (independent of upstream)
│   ├── test_deps.py                  # deps.py: lazy import + MissingExtraError
│   ├── test_server.py                # server.py: tool registration + transport selection
│   ├── test_serialization.py         # serialization.py: JSON/parquet roundtrip + file:// validation
│   ├── test_data_tools.py            # data.py: 3 loader tools
│   ├── test_split_tools.py           # split.py: 4 splitter tools
│   ├── test_eval_tools.py            # evaluate.py: 4 evaluation tools
│   ├── test_ranking_tools.py         # ranking.py: get_top_k tool
│   ├── test_state.py                 # state.py: put/get df + model, TTL, version check
│   ├── test_auth.py                  # auth.py: token verify + extract_bearer
│   ├── test_http_transport.py        # http_transport.py: /health, /invoke, 401/404
│   ├── test_script_help.py           # 6 scripts: --help exits 0
│   ├── test_smoke_movielens.py       # SAR 100k baseline alignment (nightly, @notebooks)
│   ├── test_smoke_ncf_movielens.py   # NCF smoke (nightly, @gpu)
│   └── test_smoke_sasrec_amazon.py   # SASRec smoke (nightly, @gpu)
│
└── docs/                             # Documentation
    ├── ARCHITECTURE.md               # Architecture decisions (implementation-level)
    ├── IMPLEMENTATION_PLAN.md        # Planner phases + real symbol references
    ├── tools_reference.md            # 12 tool signatures + JSON examples + error codes
    ├── usage_examples.md             # 5 agent conversation flows
    └── CODEMAP.md                    # This file
```

---

## Module dependency graph / 模块依赖关系

```
                    ┌─────────────────┐
                    │  server.py      │  Entry point
                    │  main()         │  selects stdio | http
                    └───────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
    │ tools/      │ │ http_       │ │ auth.py          │
    │ data.py     │ │ transport.py│ │ (HTTP only)      │
    │ split.py    │ │ FastAPI ASGI│ │ verify_token()   │
    │ evaluate.py │ │ /invoke     │ │ extract_bearer() │
    │ ranking.py  │ └──────┬──────┘ └──────────────────┘
    └──────┬──────┘        │
           │               │
           ▼               │
    ┌─────────────┐        │
    │ deps.py     │◄───────┘  (tools call deps for lazy imports)
    │ load_*()    │
    │ @lru_cache  │
    └──────┬──────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ recommenders.*               │  Upstream package
    │ datasets.{movielens,criteo,  │  (heavy — lazy imported)
    │   mind,python_splitters}     │
    │ evaluation.python_evaluation │
    │ models.{sar,ncf,sasrec,...}  │
    └──────────────────────────────┘

    ┌─────────────────┐
    │ state.py        │  Independent module
    │ StateStore      │  - cloudpickle for models
    │ put_df/model    │  - parquet for DataFrames
    │ get_df/model    │  - meta.json + TTL
    │ FileLock        │  - secrets.token_hex(16) handles
    └─────────────────┘

    ┌─────────────────┐
    │ serialization.py│  Called by tools/* for I/O
    │ df_to_json()    │  - JSON orient=split
    │ df_from_json()  │  - file:// parquet/json
    │ maybe_cache()   │  - path traversal guard
    │ _validate_file_ │  - MCP_FILE_ROOTS allowlist
    │   path()        │
    └─────────────────┘
```

### Key dependency rules / 关键依赖规则

1. **`deps.py` is the single import boundary** — it is the ONLY module that directly `import recommenders.*`. All tools call `deps.load_*()` functions. If upstream APIs drift, only `deps.py` needs updating.

2. **`tools/*.py` → `deps.py` + `serialization.py`** — tools never import `recommenders` directly; they go through deps for lazy loading and structured error handling.

3. **`state.py` is independent** — it does not import from `tools/` or `deps.py`. Scripts call `state.put_model()` / `state.get_model()` directly. `state.py` imports `cloudpickle` (not stdlib `pickle`) for robust model serialization across Python/library versions.

4. **`auth.py` and `http_transport.py` are HTTP-only** — stdio mode never loads them. `auth.py` is pure functions (no I/O beyond env vars). `http_transport.py` imports `auth.py` and depends on `fastapi` + `uvicorn`.

5. **`serialization.py` enforces path safety** — `_validate_file_path()` rejects `file://` URIs outside allowed roots (`cwd`, tempdir, `MCP_FILE_ROOTS`). Process-generated files (tracked in `_GENERATED_FILE_PATHS`) are always trusted.

6. **`skill/scripts/*.py` import `recommenders` directly** — they are standalone OS processes, not part of the MCP server. They may also call `mcp_server.state.StateStore` for artifact persistence and `mcp_server.deps.load_model_class` for structured error handling.

---

## Data flow summary / 数据流概要

```
Agent (natural language)
   │
   ├─[MCP stdio/HTTP]──► load_movielens ──► DataFrame JSON or file:// URI
   │                           │
   │                           └─(large df + cache_path)──► parquet on disk
   │
   ├─[MCP]──► split_random(data, ratio=0.75) ──► {train, test}
   │
   ├─[Bash]──► python skill/scripts/sar_movielens.py --model-out
   │              │
   │              ├─ import recommenders.models.sar.SARSingleNode
   │              ├─ SAR.fit(train) + recommend_k_items(remove_seen=True)
   │              ├─ state.put_model(sar) → MODEL_HANDLE=<32-hex>
   │              └─ print JSON metrics
   │
   ├─[MCP]──► eval_ranking(true, pred, k=10) ──► {precision, recall, ndcg, map, r_precision}
   │
   └─[Later session]──► state.get_model(handle) → recommend_k_items(new_data)
```

---

## CI test groups / CI 测试分组

Defined in `tests/test_groups.yml` (independent of upstream):

| Group | Tests | Trigger | Runtime target |
|---|---|---|---|
| `group_core_001` | deps, server, serialization, data/split/eval/ranking tools, state, auth, http_transport, script_help | PR gate | ≤ 15 min |
| `group_core_002` | smoke_movielens (SAR 100k baseline) | nightly | ≤ 35 min |
| `group_gpu_001` | smoke_ncf_movielens, smoke_sasrec_amazon | nightly (GPU runner) | ≤ 35 min |

---

## Build artifacts / 构建产物

| Artifact | Source | Contents |
|---|---|---|
| `recommenders-mcp:core` | `Dockerfile` stage `core` | python:3.11-slim + recommenders (CPU) + wrapper, non-root `appuser` |
| `recommenders-mcp:gpu` | `Dockerfile` stage `gpu` | nvidia/cuda:12.2-runtime + recommenders[gpu] + wrapper, non-root `appuser` |
| Console script | `pyproject.toml` `[project.scripts]` | `recommenders-mcp = mcp_server.server:main` |
