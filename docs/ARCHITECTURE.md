# Recommenders AI — 架构文档

> 面向 `tdd-guide` agent 的实现级架构文档。所有模块内部结构、签名、决策理由、测试要点均已给出，可直接据此 TDD。
>
> 上游输入：`DESIGN_recommenders_ai.md`（定稿）+ `docs/IMPLEMENTATION_PLAN.md`（planner 分阶段方案）。
> 代码符号名保持英文；文档语言中文为主（§9 拍板）。

---

## 0. 文档导航

- §1 系统全景（两层职责边界 + 三路径交互 + 数据流）
- §2 组件设计（逐模块决策与内部结构）
- §3 关键决策与权衡（§9 四项落点）
- §4 依赖与扩展性
- §5 安全考量
- §6 质量属性
- §7 模块实现清单（tdd-guide 按此拆 task）

---

## 1. 系统全景

### 1.1 两层职责边界

| 层 | 形态 | 跨平台 | 职责边界 | 不做什么 |
|---|---|---|---|---|
| **MCP server**（Docker 化进程） | stdio / HTTP 双传输，打进 `recommenders-mcp:core` / `:gpu` 两档镜像 | 任何 MCP 客户端 | 稳定、无状态、轻依赖（pandas/numpy/sklearn 级）的**纯函数原子工具**：数据加载、4 种划分、4 类评估、top-k 提取 | 不持有训练态模型对象；不直接 fit/predict；不透传重依赖（torch/TF/pyspark） |
| **Agent Skill** | `SKILL.md` + playbooks + scripts + snippets，遵循 Agent Skills 规范 | 任何支持 Skills 规范的平台 | **何时用** + **流程编排** + **训练脚本模板**（Bash 跑独立进程） + **notebook 漂移对齐** | 不重复实现 MCP 工具；不内嵌推理服务进程 |

**分工原则（不可越界）**：
- MCP 放**无状态纯函数**：函数入参 → 返回值，调用间无隐式状态（`state.py` 的缓存是显式 handle，不是隐藏可变状态）。
- Skill 放**有状态训练流程**：模型实例、checkpoint、torch/TF/pyspark 重依赖——以可运行脚本形式由 agent 用 Bash 在独立进程跑。训练产物通过 `state.py` 显式 handle 落盘回 MCP 层供后续推荐使用。

理由：上游 `recommenders/__init__.py` 仅暴露版本元数据、**无聚合 facade**，子模块按需 import；核心 `install_requires` 已重（cornac/lightgbm/transformers/numba/nltk），GPU/Spark 是额外大块；21 个 `examples/00_quick_start/*.ipynb` 是最佳实践载体（notebook-as-source-of-truth）。把训练塞进 MCP 会让 server 进程持有大模型对象 + 重依赖透传，违背"稳定轻原子工具"定位。

### 1.2 三路径交互模型

```
路径 A (stdio)：   agent 平台 ──[stdin/stdout]──> recommenders-mcp 容器（子进程）
路径 B (HTTP)：    agent 平台 ──[HTTP+Bearer]──> recommenders-mcp 容器（常驻监听 :8080）
路径 C (Skill 脚本)：agent ──[Bash]──> python skill/scripts/xxx.py（独立进程，可容器内或宿主）
                                          │
                                          └─> 训练完调 state.put_model → 返回 handle id
```

- **路径 A**：本地子进程，无网络暴露，**不鉴权**（§2.6 说明理由）。默认 `MCP_TRANSPORT=stdio`。
- **路径 B**：远程/多客户端共享，常驻端口，**强制 token 鉴权**（§2.6）。
- **路径 C**：训练脚本是独立 OS 进程，与 MCP server 无共享内存；训练产物通过 `state.py` 的 parquet/pickle 文件 + handle id 跨进程传递。脚本可调 MCP 工具做 load/split/eval，也可直接 import `recommenders` 子模块。

### 1.3 数据流图（文字版）

```
用户自然语言
   │
   ▼
Agent（读 SKILL.md 选 playbook + 脚本）
   │
   ├──[MCP stdio/HTTP]──> load_movielens ──> DataFrame JSON 或 file:// URI
   │                          │
   │                          └(大 df + cache_path)──> state.put_df → file://handle
   │
   ├──[MCP]──> split_chrono(df, ratio=0.8) ──> {train_json, test_json}
   │
   ├──[Bash]──> python skill/scripts/sar_movielens.py \
   │              --cache-path /data --model-out
   │           │
   │           ├─ import recommenders.models.sar.SARSingleNode
   │           ├─ SAR.fit(train_df)
   │           ├─ recommend_k_items(test, top_k=10) → pred_df
   │           ├─ state.put_model(sar_instance) → MODEL_HANDLE=<id>（落盘 pickle + meta.json）
   │           └─ print JSON metrics
   │
   ├──[MCP]──> eval_ranking(rating_true, rating_pred, k=10) ──> {precision, recall, ndcg, map, r_precision}
   │
   └──[后续会话]──> 凭 handle id → state.get_model → recommend_k_items(新数据) ──> 推荐
```

**关键流转规则**：
- 小 DataFrame（行数 ≤ `LARGE` 阈值，默认 50_000）：JSON（`orient="split"`）往返。
- 大 DataFrame（行数 > `LARGE` 且调用方提供 `cache_path`）：落盘 parquet，返回 `file://<path>` URI；后续工具识别 `file://` 直接读盘，避免 JSON 往返。
- 训练模型：绝不通过 MCP 工具入参/出参传递；只能通过 `state.put_model` 落盘 pickle + 返回 handle id 字符串。

---

## 2. 组件设计

### 2.1 `mcp_server/server.py` — 入口与生命周期

**职责**：解析 `MCP_TRANSPORT` env 选择 stdio / HTTP，构造 `Server("recommenders-mcp")`，注册 16 工具，启动事件循环。

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 传输选择 | `MCP_TRANSPORT` env 二选一，默认 `stdio` | stdio 是 MCP 客户端最通用路径；HTTP 是可选增强。env 切换比 CLI flag 更适配容器 entrypoint |
| 工具注册模式 | 每个工具模块暴露 `register_xxx_tools(server: Server)` 函数，`server.py` 顺序调用 4 个 register | 避免 server.py 膨胀成 god module；每模块高内责，tdd 可按模块独立测 |
| Server 实例 | 模块级单例 `server = Server("recommenders-mcp")` | MCP SDK 期望单 server；register 函数收 `server` 参数便于测试时注入 mock |
| 生命周期 | `main()` 同步入口 + `asyncio.run` 包 stdio；HTTP 用 `uvicorn.run` | stdio 是 async API；HTTP 走 ASGI。统一 `main()` 入口供 console script |
| 日志 | `logging.getLogger("recommenders-ai")`，`INFO` 默认 | 可观测性（§6）；不 print |

**内部结构骨架（tdd-guide 据此实现）**：
```python
# mcp_server/server.py
import logging, os, asyncio
from mcp.server import Server
from mcp_server.deps import MissingExtraError  # 用于注册时的错误边界

logger = logging.getLogger("recommenders-ai")
server = Server("recommenders-mcp")

def _register_all() -> None:
    from mcp_server.tools.data import register_data_tools
    from mcp_server.tools.split import register_split_tools
    from mcp_server.tools.evaluate import register_evaluate_tools
    from mcp_server.tools.ranking import register_ranking_tools
    from mcp_server.tools.score import register_score_tools
    from mcp_server.tools.handles import register_handle_tools
    for fn in (register_data_tools, register_split_tools,
               register_evaluate_tools, register_ranking_tools,
               register_score_tools, register_handle_tools):
        fn(server)

def main() -> None:
    logging.basicConfig(level=os.environ.get("MCP_LOG_LEVEL", "INFO"))
    _register_all()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        from mcp.server.stdio import stdio_server
        async def _run():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
        asyncio.run(_run())
    elif transport == "http":
        from mcp_server.http_transport import build_app
        import uvicorn
        uvicorn.run(build_app(server), host="0.0.0.0",
                    port=int(os.environ.get("MCP_HTTP_PORT", "8080")))
    else:
        raise ValueError(f"Unknown MCP_TRANSPORT: {transport}")

if __name__ == "__main__":
    main()
```

**测试要点**：
- `test_server_registers_sixteen_tools`：构造 mock server，调 `_register_all()`，断言 `register_tool` 被调 16 次。
- `test_main_unknown_transport_raises`：`MCP_TRANSPORT=foo` → `ValueError`。
- stdio/HTTP 实际握手放 smoke（nightly），不进 PR gate。

---

### 2.2 `mcp_server/deps.py` — lazy import + 结构化缺依赖错误

**职责**：集中管理对 `recommenders` 子模块的 lazy import；缺 extra 时抛结构化 `MissingExtraError`。

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| import 时机 | 函数内 lazy import（非模块顶部） | `recommenders` 核心依赖重（numba/cornac/nltk 启动慢）；lazy 让 server 握手快，按需付加载代价 |
| 缓存 | `@lru_cache(maxsize=None)` 包每个 loader 函数 | import 已完成则复用符号对象，避免重复 import 开销；lru_cache 天然线程安全（CPython GIL） |
| 错误类型 | 自定义 `MissingExtraError(RuntimeError)`，含 `extra`/`symbol`/`hint` 字段 + 安装命令字符串 | 结构化字段供 MCP 客户端解析；安装命令直接给用户可执行提示（`pip install 'recommenders[gpu]'` 或 `pull recommenders-mcp:gpu`） |
| 模型类加载 | 通用 `load_model_class(module_path, class_name, extra)` | Skill 脚本也复用此函数；统一错误边界 |
| 分层理由 | deps.py 是唯一允许 `import recommenders.*` 的地方 | 工具模块只调 deps 函数，不直接 import；若 upstream API 漂移，只改 deps 一处 |

**为何这样分层**：
- 把"依赖边界"和"业务逻辑"分离——`tools/*.py` 只管调度与序列化，不感知 recommenders 是否可 import。
- `MissingExtraError` 是**领域错误**而非通用 `ImportError`，让 MCP 工具能把缺依赖翻译成用户可操作的"拉哪档镜像"提示，而不是栈 traceback。
- lru_cache 让"import 一次"成为不变量，便于在测试里 monkeypatch 单次返回值。

**内部结构**：见 IMPLEMENTATION_PLAN §Phase 1 草案。补充：每个 loader 函数返回**符号字典或单一符号**，调用方按 key 取。`load_eval_api()` 返回 `{"rmse": rmse, "mae": mae, "precision_at_k": precision_at_k, "map": map, ...}`——注意 `map` 是 `python_evaluation.map`（非 `map_at_k`），对齐基准 0.110591。

**测试要点**：
- `test_missing_extra_error_message_contains_install_command`：断言消息含 `pip install 'recommenders[gpu]'` 和镜像名。
- `test_load_movielens_loader_returns_callable`：调用后返回 `load_pandas_df`。
- `test_load_model_class_missing_raises`：传入不存在的 module → `MissingExtraError`。

---

### 2.3 `mcp_server/serialization.py` — DataFrame JSON / 文件双模式

**职责**：DataFrame ↔ JSON 互转；大 DataFrame 落盘 parquet 返回 `file://` URI；反序列化识别 `file://` 直接读盘。

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| JSON orient | `orient="split"` | 保留 dtype 与 index；pandas 官方推荐的对称往返格式；`to_json/from_json` 一致 |
| 大 df 阈值 | `LARGE = 50_000` 行（常量，可调） | Movielens 100k 超过即走文件模式；阈值是经验值，避免 JSON 往返 MB 级字符串 |
| 触发条件 | `maybe_cache(df, cache_path)`：`len(df) > LARGE and cache_path` → 落盘 | 必须调用方显式提供 `cache_path`，server 不擅自写盘（最小权限） |
| 落盘格式 | parquet（`pyarrow`） | 比 JSON 小 3-5x，dtype 保真，pandas 原生往返 |
| URI scheme | `file://<abs_path>` | 明确协议前缀，反序列化端识别后读盘，否则当 JSON 字符串解析 |
| 返回结构 | `{"uri": "file:///data/abc.parquet", "rows": 100000, "schema": {...}}` | 带元数据便于 agent 决策 |
| 不直接 pickle 透传 DataFrame | 拒绝 | pickle 反序列化是任意代码执行风险（§5）；JSON/parquet 是数据格式非代码；模型 pickle 仅限 state.py 内部 handle，非工具入参 |

**为何不直接 pickle 透传 DataFrame**：
- 安全：pickle 反序列化 = 任意代码执行。MCP 工具入参是跨信任边界输入（HTTP 模式下来自远端），绝不能 pickle.load。
- 兼容：JSON/parquet 跨语言可读；pickle 绑 Python 版本与 pandas 版本。
- 可观测：JSON 可在日志/调试中检视；pickle 不可读。
- 模型 pickle 的例外：`state.py` 的 model handle 是本仓库脚本写入的本地文件（信任边界内，§5），不来自工具入参。

**内部结构骨架**：
```python
# mcp_server/serialization.py
import os, json, hashlib
from pathlib import Path
import pandas as pd

LARGE = 50_000  # 行数阈值

def df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")

def df_from_json(payload: str) -> pd.DataFrame:
    if payload.startswith("file://"):
        path = payload[len("file://"):]
        return pd.read_parquet(path) if path.endswith(".parquet") \
               else pd.read_json(path, orient="split")
    return pd.read_json(io.StringIO(payload), orient="split")

def maybe_cache(df: pd.DataFrame, cache_path: str | None) -> str:
    if cache_path and len(df) > LARGE:
        os.makedirs(cache_path, exist_ok=True)
        digest = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()[:12]
        fp = Path(cache_path) / f"{digest}.parquet"
        if not fp.exists():
            df.to_parquet(fp)
        return f"file://{fp.resolve()}"
    return df_to_json(df)
```

**测试要点**：
- `test_df_roundtrip_small_json`
- `test_maybe_cache_large_returns_file_uri`
- `test_maybe_cache_small_returns_json`
- `test_df_from_json_file_uri_reads_parquet`
- `test_no_cache_path_returns_json`

---

### 2.4 `mcp_server/state.py` — 落盘 + handle + TTL

**职责**：DataFrame（parquet）、模型对象（pickle）、元数据（meta.json）三件套；handle id 生成；TTL 清理（惰性 + 主动）。

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 存储布局 | 每 handle 一个目录：`<root>/<handle>/data.parquet` + `model.pkl` + `meta.json` | 三件套隔离不同序列化格式；meta.json 可读便于运维；删除即 rmtree handle 目录 |
| handle id | `secrets.token_hex(16)`（32 字符 hex） | 不可猜测（防 handle 枚举攻击）；与 auth token 同源随机 |
| DataFrame 存储 | parquet | 见 §2.3 |
| 模型存储 | cloudpickle | 模型对象（SARSingleNode/NCF/SASREC）含 torch/tf state，无原生 JSON 表征；cloudpickle 比 stdlib pickle 对 lambda/closure/跨版本兼容性更好。**边界声明**：model pickle 仅由本仓库 skill scripts 写入（信任边界内），非外部输入（§5） |
| meta.json 字段 | `{handle, kind: "df"\|"model", created_at, expires_at, recommends_version, schema?}` | `recommends_version` 用于 get_model 校验版本兼容（防 upstream 升级后旧 checkpoint 反序列化错） |
| TTL 默认 | 24h（`STATE_TTL_SECONDS` env 可覆盖） | 平衡"跨会话复用"与"磁盘无限堆积"；用户可调 |
| 惰性清理 | 每次 `put_*` 调 `_maybe_cleanup()`，按 `last_cleanup_time` 节流每小时扫一次 | 不开后台线程（进程模型简单）；put 是低频操作，附带清理代价可接受 |
| 主动清理 | `cleanup_expired()` 公开函数，可被外部 cron/compose sidecar 调 | 给运维显式入口；Docker compose 可加定时任务 |
| 并发 | 每操作用 `<handle>.lock` 文件（`FileLock`）+ 原子写（tmp + rename） | 多请求/多进程（HTTP 模式 + Skill 脚本进程）并发写同一 handle 时需互斥；原子写防半写状态 |
| 磁盘占用回收 | TTL 过期 + 可选 `STATE_MAX_BYTES` 配额（LRU 淘汰） | 兜底；防异常大模型撑爆磁盘 |

**为何三件套而非单文件**：
- parquet（数据）+ cloudpickle（模型）+ json（meta）格式各异，强行合并需自造容器格式；
- meta.json 人类可读，运维 `cat meta.json` 即可排查；
- 删除单 handle = `rmtree`，O(1) 且无碎片。

**内部结构骨架**：
```python
# mcp_server/state.py
import os, json, time, secrets, shutil
from pathlib import Path
from datetime import datetime, timezone
from filelock import FileLock

DEFAULT_TTL = int(os.environ.get("STATE_TTL_SECONDS", "86400"))
_last_cleanup = 0.0

class StateStore:
    def __init__(self, root: str, ttl_seconds: int = DEFAULT_TTL):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _new_handle(self) -> str: return secrets.token_hex(16)

    def put_df(self, df, kind_label="df") -> str: ...
    def get_df(self, handle: str) -> "pd.DataFrame": ...
    def put_model(self, model, recommends_version: str) -> str: ...
    def get_model(self, handle: str, expects_version: str): ...
    def _maybe_cleanup(self): ...
    def cleanup_expired(self): ...
```

**测试要点**：
- `test_put_get_df_roundtrip`
- `test_put_get_model_roundtrip_sar`：put SAR 实例 → get → `recommend_k_items` 可调。
- `test_ttl_expiry_removes_handle`：TTL=1s sleep 2s → `cleanup_expired` 后 get 抛 `StateNotFoundError`。
- `test_version_mismatch_raises`：meta `recommends_version` ≠ expects → `StateVersionError`。
- `test_handle_unparseable_id`：随机 32 hex，不可预测。
- `test_concurrent_put_same_root`：两线程并发 put 不互踩（FileLock 生效）。

---

### 2.5 `mcp_server/tools/*` — 16 工具分层

**分层**：
- `tools/data.py`：`load_movielens`、`load_criteo`、`load_mind`（3）
- `tools/split.py`：`split_random`、`split_chrono`、`split_stratified`、`split_numpy`（4）
- `tools/evaluate.py`：`eval_rating`、`eval_classification`、`eval_ranking`、`eval_beyond_accuracy`（4）
- `tools/ranking.py`：`get_top_k`（1）
- `tools/score.py`：`recommend`（1）
- `tools/handles.py`：`list_handles`、`describe_handle`、`delete_handle`（3）

**关键签名示例（`eval_ranking`）**：
```python
@server.tool()
def eval_ranking(
    rating_true: str,
    rating_pred: str,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    col_rating: str = DEFAULT_RATING_COL,
    col_prediction: str = DEFAULT_PREDICTION_COL,
    k: int = DEFAULT_K,
) -> dict:
    api = load_eval_api()
    t = df_from_json(rating_true); p = df_from_json(rating_pred)
    return {
        "precision": float(api["precision_at_k"](t, p, col_user=col_user, col_item=col_item, col_prediction=col_prediction, k=k)),
        "recall":    float(api["recall_at_k"](t, p, ..., k=k)),
        "ndcg":      float(api["ndcg_at_k"](t, p, ..., k=k)),
        "map":       float(api["map"](t, p, ..., k=k)),       # 注意是 map 不是 map_at_k
        "r_precision": float(api["r_precision_at_k"](t, p, ..., k=k)),
    }
```

**测试要点（每工具独立 AAA 单测）**：见 IMPLEMENTATION_PLAN 测试矩阵。

#### Score 工具（`recommend`）

**职责**：凭持久化 model handle 对 user DataFrame 评分，返回 top-k 推荐。

**关键签名**：
```python
@server.tool()
def recommend(
    model_handle: str,
    user_data: str,
    top_k: int = 10,
    col_user: str = DEFAULT_USER_COL,
    remove_seen: bool = True,
    cache_path: str | None = None,
) -> dict:
```

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 入参 | 仅 `model_handle: str`（不接受 pickle） | 模型对象永不跨 MCP 边界；只传 handle id 字符串（§5 安全边界） |
| 分发逻辑 | `hasattr(model, "recommend_k_items")` → SAR；`hasattr(model, "recommend_top_k_items")` → TF-IDF | duck-type 分发，无需 meta 中硬编码模型类型 |
| SAR 未知用户 | 通过 `model.user2index` 过滤不在训练集中的用户；返回 `skipped_user_count` | SAR 无法冷启动未知用户；显式报告跳过数而非静默丢弃 |
| TF-IDF 冷启动 | `recommend_top_k_items` 直接调用；`skipped_user_count=0` | TF-IDF 是 item-to-item，不需要用户历史 |
| 出参 | `{uri, rows, schema, skipped_user_count, model_handle}` | 与现有 DataFrame payload 格式一致；附加 skipped 计数与 handle 回显 |

#### Handle 生命周期工具（`list_handles` / `describe_handle` / `delete_handle`）

**职责**：列出、查看、删除 state.py 持久化的 handle。

**关键签名**：
```python
def list_handles(kind: str | None = None) -> list[dict]
def describe_handle(handle: str) -> dict
def delete_handle(handle: str) -> dict  # returns {"handle": str, "deleted": bool}
```

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| `list_handles` 清理 | 先调 `_maybe_cleanup()`（节流），再在内存中跳过已过期 handle | 避免返回过期但未清理的 handle |
| `describe_handle` 不加载模型 | 仅读 `meta.json` + `stat` 文件大小；不调 `get_model` | 快速元数据查询；不触发 pickle 反序列化 |
| `describe_handle` 无版本校验 | 只返回 `recommends_version` 字段，不校验 | 允许 agent 查看旧版本 handle 的元信息 |
| `delete_handle` 幂等 | 不存在返回 `{"deleted": False}`；存在则 `rmtree` 返回 `{"deleted": True}` | 安全重复调用 |
| handle 格式校验 | `_handle_dir` 校验 32 hex 字符 | 防止路径遍历（恶意 handle 如 `../etc`） |

### 2.5a `mcp_server/errors.py` — 类型化 HTTP 错误信封

**职责**：将领域异常映射为 HTTP 状态码 + 结构化 `{"error": str, "code": str, "details": dict}` 响应体。

**错误映射表**：

| 异常类型 | HTTP 状态码 | `code` 字符串 | `details` 字段 |
|---|---|---|---|
| `StateNotFoundError` | 410 | `"state_not_found"` | `{"handle": str}` |
| `StateVersionError` | 409 | `"state_version_mismatch"` | `{"expected": str, "found": str}` |
| `MissingExtraError` | 503 | `"missing_extra"` | `{"extra": str, "symbol": str}` |
| `ValueError` | 400 | `"bad_request"` | `{}` |
| `TypeError` | 400 | `"bad_request"` | `{}` |
| 未知异常 | 500 | `"internal_error"` | `{}` |

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| `error` 字段 | `str(exc)` | 向后兼容 v0.1 的错误格式 |
| `code` 字段 | 机器可读字符串 | agent 可据此做分支处理（如 410 → 重新训练） |
| `details` 字段 | 类型特定 dict | 结构化信息供 agent 解析，不需要 regex 提取 |
| 匹配顺序 | most-specific first；`StateVersionError` 在 `ValueError` 之前 | `StateVersionError` 是 `ValueError` 子类；isinstance 会匹配父类 |

**测试要点**：
- `test_to_response_state_not_found_returns_410`
- `test_to_response_state_version_returns_409`
- `test_to_response_missing_extra_returns_503`
- `test_to_response_value_error_returns_400`
- `test_to_response_unknown_returns_500`

---

### 2.6 `mcp_server/auth.py` — HTTP token 鉴权

**设计决策**：

| 决策 | 选择 | 理由 |
|---|---|---|
| 鉴权方式 | Bearer token，env `MCP_HTTP_TOKEN` 强制存在 | 最简可控 |
| 比较方式 | `secrets.compare_digest(token, expected)` | 常量时间比较，防时序侧信道 |
| 缺 token 启动 | `build_app` 启动时校验 env，缺失 `AuthConfigError` 拒绝启动 | fail-closed |
| stdio 不鉴权 | stdio 是本地子进程，stdin/stdout 无网络暴露面 | 鉴权增加本地复杂度且无收益 |

**测试要点**：
- `test_verify_token_correct_returns_true`
- `test_verify_token_wrong_returns_false`
- `test_missing_env_raises_auth_config_error`
- `test_http_no_token_returns_401`、`test_http_correct_token_returns_200`（集成，nightly）

---

### 2.7 `skill/` — SKILL.md + playbook + scripts + snippets

**为何训练不进 MCP**：
- 训练产生**有状态对象**（模型实例 + checkpoint），跨 MCP 调用难承载——MCP 工具是无状态纯函数。
- 训练依赖**重依赖**（torch/TF/pyspark），塞进 server 会让 core 镜像变 GB 级，违背"core 最轻"定位。
- 训练是**长时操作**，MCP 工具调用期望秒级返回；训练脚本走 Bash 独立进程。
- 训练流程随 upstream notebook 演化最快，脚本形式易跟进。

**scripts 统一结构**：
```python
"""
Source: examples/00_quick_start/sar_movielens.ipynb
依赖档: core
"""
import argparse, json
from mcp_server.deps import ...
from mcp_server.state import StateStore

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="100k")
    # ... --ratio --top-k --cache-path --model-out
    # load → split → fit → recommend → eval → print JSON
    # 可选 state.put_model → print MODEL_HANDLE=<id>
```

**测试要点**：
- `test_script_help_exits_zero`：7 脚本 `--help` 退出码 0。
- `test_sar_script_mock100_prints_metrics_json`：`--size mock100` 跑完 stdout 合法 JSON。
- `test_sar_script_model_out_prints_handle`：`--model-out` → stdout 含 `MODEL_HANDLE=`。

---

## 3. 关键决策与权衡（§9 四项落点）

| §9 决策 | 架构落点 | 代价 |
|---|---|---|
| **不做 Spark/ALS** | 只 core + gpu 两档 Dockerfile build arg；deps 无 pyspark loader；SKILL 能力矩阵 ALS 标"不支持" | 丢失分布式 ALS；省 JDK 依赖、镜像小一档、CI 简化 |
| **训练产物持久化** | state.py 三件套 + handle id + TTL 24h + 惰性+主动清理；Skill `--model-out` 调 put_model；后续会话凭 handle 取 | 磁盘需 TTL+配额兜底；pickle 需版本校验；并发需 FileLock |
| **stdio + HTTP 双传输** | server.py env 切换；http_transport ASGI；auth 强制 token；.mcp.json 两段 | HTTP 增加鉴权复杂度与部署面 |
| **中文为主** | 文档中文；代码符号/工具名/错误码英文 | 国际用户依赖英文符号；README 中英双语兜底 |

---

## 4. 依赖与扩展性

### 4.1 recommenders extras → 镜像档位映射

| extras | 镜像档 | Dockerfile build arg | 包含能力 |
|---|---|---|---|
| core | `recommenders-mcp:core` | `COMPUTE=core` | 数据/划分/评估/top-k + SAR/RLRMC/TF-IDF/LightGBM/BPR |
| `gpu` | `recommenders-mcp:gpu` | `COMPUTE=gpu` | core + torch/TF |
| `experimental` | 不单独出镜像 | 用户按需 pip | xlearn/vowpalwabbit/nni/lightfm/scikit-surprise |
| `dev` | 不进镜像 | 本地 `pip install -e .[dev]` | black/pytest |
| `spark` | **不出镜像**（§9） | — | ALS 不支持 |

### 4.2 扩展点

| 扩展场景 | 扩展位置 |
|---|---|
| 新增 MCP 工具 | `tools/<domain>.py` + `register_<domain>_tools` + `server._register_all` + `deps` loader |
| 新增训练脚本 | `skill/scripts/<name>.py` + `SKILL.md` 能力矩阵 + `test_groups.yml` |
| 新增数据集加载器 | `deps.load_<dataset>()` + `tools/data.py` |
| 新增评估指标 | `deps.load_eval_api` 字典加 key + `tools/evaluate.py` |
| 新增镜像档 | `Dockerfile` 加 `COMPUTE=<x>` 分支 + compose 服务 |

### 4.3 notebook 漂移版本对齐策略

1. 脚本顶部注释标注 `Source: examples/00_quick_start/<nb>.ipynb`。
2. deps.py 集中 import，符号漂移只改一处。
3. CI smoke 对齐 `tests/smoke/examples/test_notebooks_python.py:25-28` 基准（Precision≈0.330753, nDCG≈0.382461, Recall≈0.176385, MAP≈0.110591，TOL=0.05）。
4. meta.json 记录 `recommends_version`，不匹配抛 `StateVersionError`。
5. wrapper 锁 `recommenders>=1.2.1,<2`。

---

## 5. 安全考量

### 5.1 HTTP 鉴权
- HTTP 强制 `MCP_HTTP_TOKEN` env，缺失拒绝启动（fail-closed）。
- `secrets.compare_digest` 常量时间比较。
- 401 响应只 `"unauthorized"`，不泄露期望值。
- stdio 不鉴权（信任边界=本地进程启动权限）。

### 5.2 pickle 反序列化边界声明

| pickle 来源 | 是否信任 | 理由 |
|---|---|---|
| `state.put_model` 写的 model.pkl（本仓库脚本写） | 信任（边界内） | 写入方是本仓库脚本调 recommenders.models.* 实例，非外部输入 |
| MCP 工具入参的 DataFrame | **不**信任，禁 pickle | HTTP 下入参来自远端，pickle.load=任意代码执行；强制 JSON/parquet |
| 外部用户上传 model.pkl | **不**信任 | get_model 只读 state.root，不读任意路径 |
| `recommend` 工具的 `model_handle` 入参 | 仅接受 handle id 字符串 | **模型对象永不跨 MCP 边界**；`recommend` 内部调 `store.get_model` 加载模型，评分后丢弃引用 |

### 5.3 state 目录权限
- `state.root` 由进程 owner 独占可写；compose 挂独立 volume。
- handle id 用 `secrets.token_hex(16)` 不可枚举。
- meta.json 不存敏感数据。

### 5.4 其他
- 无 hardcoded secrets。
- 工具入参走 `df_from_json` 校验，非法 JSON 抛错（输入验证边界）。
- `file://` URI 经 `_validate_file_path` 校验：仅允许 `cwd`、系统临时目录、`MCP_FILE_ROOTS` 环境变量指定的路径，且必须绝对路径；本进程通过 `maybe_cache` 产生的文件（`_GENERATED_FILE_PATHS` 集合）始终信任。
- 镜像不内嵌凭证；token 走 env。

---

## 6. 质量属性

### 6.1 可测性
- 每工具可独立单测，纯函数入参 JSON 字符串，mock deps loader 即可隔离。
- AAA + `pytest.approx`。
- `deps.lru_cache` 便于 monkeypatch。
- PR gate（CPU core 单测）：data/split/eval/ranking/score/handle/state/errors，~64 用例，≤15min。
- nightly：smoke SAR/TF-IDF custom + notebook 对齐基准（7 smoke）。

### 6.2 可观测性
- `logging.getLogger("recommenders-ai")`，`MCP_LOG_LEVEL` 可调。
- 每工具 log 入参摘要（行数/列名）+ 返回摘要。
- state 清理 log 删除的 handle。
- deps lazy import 成功/失败 log。
- HTTP 请求 log method/path/状态码（不 log token）。

### 6.3 性能
- 大 df（>50_000 + cache_path）走 parquet `file://`，避免 JSON MB 级往返。
- `lru_cache` 复用 import。
- `_maybe_cleanup` 每小时节流。
- 并发 FileLock 不长持。
- Skill 脚本独立进程，训练不阻塞 server。

### 6.4 可维护性
- 小文件分层：每工具模块 <300 行；server.py <100 行。
- 单一 import 边界（deps.py）。
- 文档中文 + 代码英文符号。
- 扩展点清晰，新增不改现有。
- Dockerfile 多阶段 + build arg，core/gpu 共享 base。

---

## 7. 模块实现清单（tdd-guide 按 Phase 拆 task）

| Phase | 模块 | 关键交付 | 测试 |
|---|---|---|---|
| 1 | `pyproject.toml`, `server.py`, `deps.py`, `schemas.py`, `state.py`(骨架), `serialization.py`(骨架), `__init__.py` | 可安装 + 空 server 握手 + MissingExtraError | `test_deps.py`, `test_server_registers` |
| 2 | `skill/SKILL.md`, `playbooks/01..05.md` | 能力矩阵 + 5 playbook 互链 | 文档完备性 |
| 3 | `serialization.py`(完整), `tools/{data,split,evaluate,ranking}.py`, `tools/__init__.py` | 12 工具可调 | `test_{data,split,eval,ranking}_tools.py` |
| 4 | `skill/scripts/*.py`(7), `skill/snippets/*.md` | 7 脚本 `--help` + 跑通 | `test_script_help`, `test_sar_script_mock100` |
| 5 | `tests/conftest.py`, `test_state.py`, `test_smoke_movielens.py`, `test_groups.yml`, pytest ini | 单测 + smoke 对齐基准 | 覆盖率 ≥80% |
| 6 | `state.py`(完整), `http_transport.py`, `auth.py`, `Dockerfile`, `docker-compose.yml`, `.mcp.json` | 两档镜像 + HTTP 鉴权 + state 往返 | `test_state.py`, `test_http_auth.py` |
| 7 | `README.md`, `docs/tools_reference.md`, `docs/usage_examples.md` | 中英双语 + 12 工具 reference + 5 对话示例 | 文档验收 |
| 8 | `tools/score.py`, `tools/handles.py`, `errors.py`, `skill/scripts/tfidf_custom.py`, `playbooks/06_recommend.md` | recommend 工具 + handle 生命周期 + 类型化错误信封 + TF-IDF 冷启动脚本 + playbook | `test_score_tools.py`, `test_handle_tools.py`, `test_errors.py`, `test_smoke_tfidf_custom.py` |

**tdd-guide 执行顺序**：Phase 1 → 3（serialization 完整先行）→ 6（state 完整）→ 4（脚本依赖 state）→ 5 → 2 → 7。每 Phase 内 RED→GREEN→IMPROVE。

---

## 附：关键不变量（tdd-guide 必须用测试守护）

1. MCP 工具入参绝不接受 pickle（serialization 只走 JSON/parquet/file://）。
2. `state.get_model` 必须校验 `recommends_version` 一致。
3. HTTP 模式缺 `MCP_HTTP_TOKEN` 必须拒绝启动。
4. `eval_ranking` 必须用 `map` 非 `map_at_k`（对齐基准）。
5. `split_random` 必须处理 list 返回（取 `[0]/[1]`）。
6. handle id 必须 `secrets.token_hex`（不可预测）。
7. `state._maybe_cleanup` 每小时节流，不每次操作扫盘。
8. skill 脚本必须顶部标注 `Source: <notebook path>`。
9. `recommend` 工具只接受 `model_handle: str`，模型对象永不跨 MCP 边界传递。
10. `errors.to_response` 的异常匹配必须 most-specific first（`StateVersionError` 在 `ValueError` 之前）。
