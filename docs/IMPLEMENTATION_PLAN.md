# Implementation Plan: Recommenders AI Wrapper (MCP Server + Agent Skill)

> 来源：planner agent 基于 `D:\claudecode\Recommenders\DESIGN_recommenders_ai.md` 探查真实仓库后产出。
> 落盘目标：本文件。供 architect / tdd-guide / e2e-runner / code-reviewer / doc-updater 后续 agent 直接引用。

## Overview

把上游 `recommenders` v1.2.1（`D:\claudecode\Recommenders`，`recommenders/__init__.py` 中 `__version__="1.2.1"`）封装为独立 wrapper 仓库 `recommenders-ai/`，包含两层：Docker 化 MCP server（12 个轻原子工具，core+gpu 两档镜像，不做 spark）+ 跨平台 Agent Skill（SKILL.md + 5 playbook + 6 训练脚本模板 + snippets）。训练产物由 `state.py` 落盘 checkpoint + handle id + TTL 清理。传输 stdio + HTTP（后者加 token 鉴权）。

## Requirements

- MCP 12 工具覆盖：3 数据加载 / 4 划分 / 4 评估 / 1 top-k，全部 core 轻依赖，映射真实符号。
- 训练类（SAR/NCF/SASRec/LightGBM/TF-IDF 等）不进 MCP，由 Skill 脚本调用。
- 两档镜像：`recommenders-mcp:core`、`recommenders-mcp:gpu`，无 spark 变体。
- 训练产物跨会话持久化：`state.py` 落盘 checkpoint + handle id + TTL。
- stdio + HTTP 双传输，HTTP 加 token 鉴权。
- Python 3.11，black，pytest marker（none/spark/gpu）。
- Smoke 基准对齐 SAR Movielens 100k：Precision@10≈0.330753, nDCG@10≈0.382461, Recall@10≈0.176385, MAP≈0.110591（来自 `tests/smoke/examples/test_notebooks_python.py:25-28`，TOL=0.05）。

## 真实符号依据（已探查）

- `recommenders/datasets/movielens.py:148` `load_pandas_df(size="100k", header=None, local_cache_path=None, title_col=None, genres_col=None, year_col=None)`，size 取值 `("100k","1m","10m","20m","mock100")`。
- `recommenders/datasets/criteo.py:34` `load_pandas_df(size="sample", local_cache_path=None, header=DEFAULT_HEADER)`。
- `recommenders/datasets/mind.py:62` `download_mind(size="small", dest_path=None) -> (train_path, valid_path)`；`extract_mind(train_zip, valid_zip, train_folder="train", valid_folder="valid", clean_zip_file=True) -> (train_path, valid_path)`；size 取值 `["small","large","demo"]`。
- `recommenders/datasets/python_splitters.py`：`python_random_split(data, ratio=0.75, seed=42)`、`python_chrono_split(data, ratio=0.75, min_rating=1, filter_by="user", col_user, col_item, col_timestamp)`、`python_stratified_split(data, ratio=0.75, min_rating=1, filter_by="user", col_user, col_item, seed=42)`、`numpy_stratified_split(X, ratio=0.75, seed=42)`。注意 `python_random_split` 返回 list/DataFrame，`python_chrono_split/stratified` 返回 list。
- `recommenders/evaluation/python_evaluation.py`：rating `rmse/mae/rsquared/exp_var`（行 165/198/231/264，签名 `rating_true, rating_pred, col_user, col_item, col_rating, col_prediction`）；classification `auc/logloss`（行 297/340，同签名）；ranking `precision_at_k/recall_at_k/r_precision_at_k/ndcg_at_k/map_at_k`（行 457/510/557/616/809，签名 `rating_true, rating_pred, col_user, col_item, col_prediction, relevancy_method="top_k", k=DEFAULT_K=10, threshold=DEFAULT_THRESHOLD=10, **_`；`ndcg_at_k` 额外 `score_type="binary", discfun_type="loge"`；`map` 行 751，`map_at_k` 行 809）；beyond-accuracy `diversity(1342), novelty(1439), serendipity(1632), catalog_coverage(1680), distributional_coverage(1714)`。
- `get_top_k_items(dataframe, col_user, col_rating, k=DEFAULT_K)` 行 866，加 `rank` 列。
- `recommenders/utils/constants.py`：`DEFAULT_USER_COL="userID"`, `DEFAULT_ITEM_COL="itemID"`, `DEFAULT_RATING_COL="rating"`, `DEFAULT_PREDICTION_COL="prediction"`, `DEFAULT_TIMESTAMP_COL="timestamp"`, `DEFAULT_K=10`, `DEFAULT_THRESHOLD=10`, `SEED=42`。
- `recommenders/models/sar/sar_singlenode.py:34` `SARSingleNode(col_user, col_item, col_rating, col_timestamp, col_prediction, similarity_type=SIM_JACCARD, time_decay_coefficient=30, time_now=None, timedecay_formula=False, threshold=1, normalize=False)`，`fit(df)` 行 226，`recommend_k_items(test, top_k=10, sort_top_k=True, remove_seen=False)` 行 522。
- `recommenders/models/ncf/ncf_singlenode.py:17` `NCF(n_users, n_items, model_type="NeuMF", n_factors=8, layer_sizes=[16,8,4], n_epochs=50, batch_size=64, learning_rate=5e-3, verbose=1, seed=None)`，`fit(data)` 行 267，`predict(user_input, item_input, is_list=False)` 行 326。
- `recommenders/models/sasrec/model.py:390` `SASREC(**kwargs)`（item_num/seq_max_len/num_blocks/embedding_dim/...），`predict(inputs)` 行 564。
- `recommenders/models/tfidf/tfidf_utils.py:17` `TfidfRecommender`，`fit(tf, vectors_tokenized)` 行 201，`recommend_top_k_items(df_clean, k=5)` 行 299。
- `recommenders/models/lightgbm/lightgbm_utils.py:27` `NumEncoder`。
- `setup.py` extras：`gpu`(tensorflow/torch/nvidia-ml-py), `spark`(pyspark), `dev`(black/pytest/pytest-cov/pytest-mock), `experimental`(xlearn/vowpalwabbit/nni/lightfm/scikit-surprise)，`all` 聚合。
- `examples/00_quick_start/` 21 个 notebook：`sar_movielens.ipynb`、`ncf_movielens.ipynb`、`sasrec_amazon.ipynb`、`lightgbm_tinycriteo.ipynb`、`tfidf_covid.ipynb`、`lightgbm_movielens.ipynb` 等。

---

## Architecture Changes（wrapper 仓库 `recommenders-ai/` 全新建）

| 路径 | 职责 |
|---|---|
| `recommenders-ai/pyproject.toml` | 依赖 `recommenders` + `mcp` SDK；console script 入口 `recommenders-mcp`；锁定 Python 3.11 |
| `recommenders-ai/Dockerfile` | 多阶段，`COMPUTE=core\|gpu` build arg，对应 `setup.py` extras |
| `recommenders-ai/docker-compose.yml` | stdio 示例 + HTTP 服务 + 数据卷挂载 |
| `recommenders-ai/.mcp.json` | stdio + HTTP 客户端配置样例 |
| `recommenders-ai/mcp_server/server.py` | stdio/HTTP MCP server 入口，注册 12 工具 |
| `recommenders-ai/mcp_server/tools/{data,split,evaluate,ranking}.py` | 12 工具实现 |
| `recommenders-ai/mcp_server/schemas.py` | typed input/output dataclass |
| `recommenders-ai/mcp_server/deps.py` | lazy import + 缺失 extra 友好报错 |
| `recommenders-ai/mcp_server/state.py` | 数据集 df 缓存 + 训练 checkpoint 落盘 + handle id + TTL |
| `recommenders-ai/mcp_server/serialization.py` | DataFrame JSON 序列化/反序列化 + cache_path 文件模式 |
| `recommenders-ai/mcp_server/auth.py` | HTTP token 鉴权中间件 |
| `recommenders-ai/skill/SKILL.md` | 何时用 / 能力矩阵 / 5 步 playbook / 依赖档位表 |
| `recommenders-ai/skill/playbooks/0[1-5]_*.md` | 5 个 playbook |
| `recommenders-ai/skill/scripts/*.py` | 6 个训练脚本模板 |
| `recommenders-ai/skill/snippets/*.md` | 短代码片段 |
| `recommenders-ai/tests/test_{data,split,eval,smoke_movielens}_tools.py` | 单测 + smoke |
| `recommenders-ai/tests/test_groups.yml` | wrapper 自己的 CI 分组（不污染 upstream） |
| `recommenders-ai/docs/{tools_reference,usage_examples}.md` | 工具参考 + agent 调用示例 |
| `recommenders-ai/README.md` | 中英双语安装/使用 |

---

## Implementation Steps

### Phase 1: 范围 + 骨架（pyproject + 空 server + deps）

**目标**：仓库可安装、空 MCP server 可启动、lazy import 模式确立。

**文件清单**：
- `recommenders-ai/pyproject.toml`
- `recommenders-ai/mcp_server/__init__.py`
- `recommenders-ai/mcp_server/server.py`
- `recommenders-ai/mcp_server/deps.py`
- `recommenders-ai/mcp_server/schemas.py`
- `recommenders-ai/mcp_server/state.py`（占位骨架）
- `recommenders-ai/mcp_server/serialization.py`（占位骨架）
- `recommenders-ai/README.md`（最小）

**接口/签名草案**：

`pyproject.toml`：
```toml
[project]
name = "recommenders-ai"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "recommenders>=1.2.1,<2",
  "mcp>=1.0",
  "pydantic>=2",
  "fastapi>=0.110",
  "uvicorn>=0.27",
  "pandas>=2.0,<3",
  "pyarrow>=10",
]
[project.optional-dependencies]
dev = ["black>=23", "pytest>=7", "pytest-cov>=4", "pytest-mock>=3"]
[project.scripts]
recommenders-mcp = "mcp_server.server:main"
```

`mcp_server/deps.py`（lazy import + 缺失 extra 友好报错）：
```python
from functools import lru_cache
from typing import Any

class MissingExtraError(RuntimeError):
    def __init__(self, extra: str, symbol: str, hint: str = ""):
        self.extra = extra
        self.symbol = symbol
        super().__init__(
            f"Symbol '{symbol}' requires extra '{extra}'. "
            f"Install with: pip install 'recommenders-ai[{extra}]' "
            f"or pull recommenders-mcp:{extra} image. {hint}"
        )

@lru_cache(maxsize=None)
def load_movielens_loader():
    from recommenders.datasets.movielens import load_pandas_df
    return load_pandas_df

@lru_cache(maxsize=None)
def load_criteo_loader():
    from recommenders.datasets.criteo import load_pandas_df
    return load_pandas_df

@lru_cache(maxsize=None)
def load_mind_api():
    from recommenders.datasets.mind import download_mind, extract_mind
    return download_mind, extract_mind

@lru_cache(maxsize=None)
def load_splitters():
    from recommenders.datasets.python_splitters import (
        python_random_split, python_chrono_split,
        python_stratified_split, numpy_stratified_split,
    )
    return {"random": python_random_split, "chrono": python_chrono_split,
            "stratified": python_stratified_split, "numpy": numpy_stratified_split}

@lru_cache(maxsize=None)
def load_eval_api():
    from recommenders.evaluation.python_evaluation import (
        rmse, mae, rsquared, exp_var, auc, logloss,
        precision_at_k, recall_at_k, r_precision_at_k, ndcg_at_k, map, map_at_k,
        diversity, novelty, serendipity, catalog_coverage, distributional_coverage,
        get_top_k_items,
    )
    return {"rmse": rmse, "mae": mae, ...}

def load_model_class(module_path: str, class_name: str, extra: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError as e:
        raise MissingExtraError(extra, f"{module_path}.{class_name}", str(e)) from e
```

`mcp_server/server.py`（空骨架）：
```python
import logging, os, asyncio
from mcp.server import Server
from mcp_server.deps import MissingExtraError

logger = logging.getLogger("recommenders-ai")
server = Server("recommenders-mcp")

def main() -> None:
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

`schemas.py`、`state.py`、`serialization.py` 占位接口见下方各 Phase。

**验收标准**：`pip install -e .[dev]` 成功；`recommenders-mcp` 可启动握手；`deps.load_movielens_loader()` 返回函数对象；`MissingExtraError` 抛出含安装命令；`python -c "import mcp_server.server"` 不报错。

**依赖关系**：无（起点）。

---

### Phase 2: Skill 骨架（SKILL.md + 5 playbook）

**目标**：Agent 可读 SKILL.md 知道何时用、能选对 playbook、查到依赖档位表。

**文件清单**：
- `recommenders-ai/skill/SKILL.md`
- `recommenders-ai/skill/playbooks/01_prepare_data.md` ~ `05_optimize.md`

**SKILL.md 结构**：frontmatter（name/description）+ 何时用 + 能力矩阵（6 脚本 × MCP 工具 × 镜像档）+ 5 步 playbook 索引 + 缺依赖处理 + 训练产物 handle 说明。

**验收标准**：5 playbook 互相链接；能力矩阵列出全部 6 脚本 + 12 工具对应关系；缺依赖段含镜像档位提示。

**依赖关系**：Phase 1。

---

### Phase 3: MCP 工具（12 个）

**目标**：12 工具全部实现并可独立调用。

**文件清单**：`serialization.py`（完整）、`tools/data.py`、`tools/split.py`、`tools/evaluate.py`、`tools/ranking.py`、`tools/__init__.py`、`server.py`（注册）。

**serialization.py**：`df_to_json`（orient=split）、`df_from_json`（支持 `file://` URI + parquet/csv/json）、`maybe_cache`（行数>LARGE 且有 cache_path 时落盘 parquet 返回 `file://` URI）。

**12 工具签名**：见设计文档 §4 与上方草案。关键点：
- `split_random/chrono/stratified` 统一 `splits[0]/splits[1]`，处理 list 返回。
- `eval_ranking` 用 `map`（非 `map_at_k`），对齐基准 0.110591。
- `eval_beyond_accuracy` 接 `train_df, reco_df`。
- `get_top_k` 返回含 `rank` 列的 DataFrame。

**验收标准**：12 工具可枚举调用；`load_movielens(size="mock100")` 非空；`split_random(ratio=0.5)` 两半无交集；`eval_ranking` 返回 5 float；`get_top_k` 含 `rank` 列；大 df+cache_path 返回 `file://` URI。

**依赖关系**：Phase 1。

---

### Phase 4: 训练脚本模板（6 个）

**目标**：改写 `examples/00_quick_start/*.ipynb` 为可执行脚本，参数化入口。

**文件清单**：`skill/scripts/{sar_movielens,ncf_movielens,sasrec_amazon,lightgbm_tinycriteo,tfidf_covid,eval_quickstart}.py` + `skill/snippets/*.md`。

**统一结构**：argparse 入口（--size/--ratio/--top-k/--cache-path/--model-out）→ load → split → fit → recommend_k_items → eval → print JSON metrics → 可选 `state.put_model` 落盘返回 handle。脚本顶部注释标注来源 notebook。

**验收标准**：6 脚本 `--help` 完整；`sar_movielens.py --size mock100` 跑完打印 JSON；`--model-out` 输出 `MODEL_HANDLE=<id>`。

**依赖关系**：Phase 3（可选调 MCP）+ Phase 6（state.put_model 完整）。

---

### Phase 5: 测试（单测 + smoke）

**目标**：每工具 AAA 单测 + 端到端 smoke 对齐 SAR 100k 基准。

**文件清单**：`tests/conftest.py`、`test_{data,split,eval,ranking,state}_tools.py`、`test_smoke_movielens.py`、`test_groups.yml`、pyproject `[tool.pytest.ini_options]`。

**测试矩阵**：见下方汇总表。smoke 对齐 4 指标 TOL=0.05。

**验收标准**：CPU core 全单测过；smoke 4 指标 TOL 内；`--collect-only` 列全；覆盖率 >= 80%。

**依赖关系**：Phase 3 + 4。

---

### Phase 6: Docker 化 + state.py + HTTP 鉴权

**目标**：两档镜像可构建；docker-compose 拉起；state 落盘 checkpoint+handle+TTL；HTTP token 鉴权。

**文件清单**：`state.py`（完整）、`http_transport.py`、`auth.py`、`Dockerfile`、`docker-compose.yml`、`.mcp.json`、`tests/test_state.py`。

**state.py**：`StateStore`（put_df/get_df parquet + put_model/get_model pickle + meta.json + TTL _maybe_cleanup 每小时扫）。

**auth.py**：`verify_token` 强制 `MCP_HTTP_TOKEN` env，`secrets.compare_digest`。

**Dockerfile**：多阶段 `COMPUTE=core|gpu` build arg；gpu base `nvidia/cuda`。

**docker-compose**：stdio + http 两服务，数据卷 + state 卷。

**.mcp.json**：stdio（docker run -i）+ http（url + Bearer header）。

**验收标准**：core/gpu 镜像可构建；http 无 token 401、正确 token 200；state model 往返可 `recommend_k_items`；TTL 过期清理。

**依赖关系**：Phase 1 + 4。

---

### Phase 7: 文档（README + tools_reference + usage_examples）

**目标**：中英双语，agent 与人即可上手。

**文件清单**：`README.md`、`docs/tools_reference.md`（12 工具签名+示例+错误码）、`docs/usage_examples.md`（5 对话示例）。

**验收标准**：12 工具 reference 齐全；README 中英双语含安装/Docker/.mcp.json/MIT 署名。

**依赖关系**：Phase 3 + 6。

---

## 风险与缓解落点

| 风险 | 缓解落点 |
|---|---|
| 镜像体积（gpu GB 级） | Dockerfile 多阶段 + COMPUTE build arg；core 最轻 |
| GPU 透传 | gpu base `nvidia/cuda`；compose `deploy.resources.reservations.devices`；无卡只 core |
| notebook 漂移 | 脚本顶部标注来源 notebook；smoke 对齐 `tests/smoke/examples/test_notebooks_python.py` 基准 |
| DataFrame JSON 序列化开销 | `serialization.maybe_cache` 行数>LARGE+cache_path 落盘 parquet 返回 `file://` |
| 训练产物无限堆积 | `state.py` TTL 默认 24h + `_maybe_cleanup` 每小时扫 |
| HTTP 未授权访问 | `auth.verify_token` 强制 env + `secrets.compare_digest` |
| checkpoint 反序列化版本不兼容 | state meta 记录 `recommenders.__version__`，get_model 校验 |
| `python_chrono_split` 无 seed | 工具签名保留 seed 但文档标注"chrono 非随机" |
| `python_random_split` 返回 list/DataFrame 不一致 | `split.py` 统一 `splits[0]/splits[1]` |
| `map` vs `map_at_k` 混淆 | `eval_ranking` 用 `map` 对齐基准 0.110591 |

---

## 测试矩阵汇总

| 测试文件 | 用例数 | pytest marker | CI 分组 | 触发 |
|---|---|---|---|---|
| test_data_tools.py | 4 | (none) | group_core_001 | PR gate |
| test_split_tools.py | 4 | (none) | group_core_001 | PR gate |
| test_eval_tools.py | 4 | (none) | group_core_001 | PR gate |
| test_ranking_tools.py | 2 | (none) | group_core_001 | PR gate |
| test_state.py | 4 | (none) | group_core_001 | PR gate |
| test_smoke_movielens.py | 1 | `notebooks` | group_core_002 | nightly |
| test_smoke_ncf_movielens.py | 1 | `gpu` | group_gpu_001 | nightly |
| test_smoke_sasrec_amazon.py | 1 | `gpu` | group_gpu_001 | nightly |

**test_groups.yml 归属**：wrapper 自有 `recommenders-ai/tests/test_groups.yml`，与 upstream 完全独立。

---

## 成功标准

- [ ] `pip install -e .[dev]` 成功，`recommenders-mcp` 可启动
- [ ] 12 MCP 工具均可被客户端枚举与调用
- [ ] `deps.MissingExtraError` 缺 extra 时返回结构化提示
- [ ] `state.py` put/get model 往返可 `recommend_k_items`，TTL 生效
- [ ] `docker build --build-arg COMPUTE=core/gpu .` 均成功
- [ ] HTTP 无 token 401，正确 token 200
- [ ] smoke SAR Movielens 100k 四指标 TOL=0.05 内对齐基准
- [ ] CPU core 全单测过，覆盖率 >= 80%
- [ ] 6 训练脚本 `--help` 可执行
- [ ] SKILL.md + 5 playbook + 12 工具 reference + 5 对话示例齐全
- [ ] README 中英双语含安装、Docker、.mcp.json、MIT 署名
