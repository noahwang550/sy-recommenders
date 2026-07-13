# Usage Examples / 使用示例

Five end-to-end agent conversations demonstrating the full pipeline: natural language → MCP tool call → Skill script → metrics returned. Each example shows how an agent orchestrates the MCP tools and training scripts.

五个端到端 agent 对话示例，演示完整流程：自然语言 → MCP 工具调用 → Skill 脚本 → 指标回传。每个示例展示 agent 如何编排 MCP 工具与训练脚本。

---

## 1. SAR Movielens — 从数据加载到排名评估

**Agent 场景**：用户说"用 SAR 在 Movielens 100k 上跑一遍，给我看排名指标"。

### Step 1 — 加载数据

```json
// Agent 调用
{"tool": "load_movielens", "arguments": {"size": "100k", "cache_path": "/tmp/recommenders"}}

// Server 返回（>50k 行 → file:// 模式）
{
  "uri": "file:///tmp/recommenders/a1b2c3d4e5f6.parquet",
  "rows": 100000,
  "schema": {"userID": "int64", "itemID": "int64", "rating": "float64", "timestamp": "int64"}
}
```

### Step 2 — 随机划分

```json
// Agent 调用
{
  "tool": "split_random",
  "arguments": {
    "data": "file:///tmp/recommenders/a1b2c3d4e5f6.parquet",
    "ratio": 0.75,
    "seed": 42
  }
}

// Server 返回
{
  "train": {"uri": "file:///tmp/recommenders/b2c3d4e5f6a7.parquet", "rows": 75000, "schema": {}},
  "test":  {"uri": "file:///tmp/recommenders/c3d4e5f6a7b8.parquet", "rows": 25000, "schema": {}}
}
```

### Step 3 — 训练（Skill 脚本，独立进程）

```bash
python skill/scripts/sar_movielens.py --size 100k --model-out --state-root /app/state
```

输出：

```json
{
  "precision": 0.330753,
  "recall": 0.176385,
  "ndcg": 0.382461,
  "map": 0.110591
}
MODEL_HANDLE=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
```

### Step 4 — 评估排名

```json
// Agent 调用
{
  "tool": "eval_ranking",
  "arguments": {
    "rating_true": "file:///tmp/recommenders/c3d4e5f6a7b8.parquet",
    "rating_pred": "<sar_prediction_df_json>",
    "k": 10
  }
}

// Server 返回
{
  "precision": 0.330753,
  "recall": 0.176385,
  "ndcg": 0.382461,
  "map": 0.110591,
  "r_precision": 0.247
}
```

### Step 5 — 后续会话复用模型

```python
from mcp_server.state import StateStore

store = StateStore("/app/state")
model = store.get_model("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6")
predictions = model.recommend_k_items(new_test_df, top_k=10, remove_seen=True)
```

---

## 2. NCF Movielens — GPU 模型训练

**Agent 场景**：用户说"我想试试 NCF 神经网络推荐器"。

### Step 1 — 检查镜像档位

NCF 需要 GPU extra。如果 agent 运行在 `core` 镜像中，训练脚本会抛 `MissingExtraError`：

```bash
python skill/scripts/ncf_movielens.py --size 100k --epochs 5
```

错误输出（结构化提示）：

```
MissingExtraError: Symbol 'recommenders.models.ncf.ncf_singlenode.NCF' requires extra 'gpu'.
Install with: pip install 'recommenders-ai[gpu]' or pull recommenders-mcp:gpu image.
```

**Agent 动作**：切换到 `recommenders-mcp:gpu` 镜像后重新执行。

### Step 2 — GPU 镜像中训练

```bash
python skill/scripts/ncf_movielens.py --size 100k --epochs 5 --model-out --state-root /app/state
```

输出：

```json
{
  "precision": 0.28,
  "recall": 0.15,
  "ndcg": 0.32,
  "map": 0.09
}
MODEL_HANDLE=f1e2d3c4b5a6f7e8d9c0b1a2f3e4d5c6
```

### Step 3 — 与 SAR 对比评估

```json
// Agent 对 NCF 和 SAR 的预测分别调用 eval_ranking
{"tool": "eval_ranking", "arguments": {"rating_true": "<test_uri>", "rating_pred": "<ncf_pred_uri>", "k": 10}}
// → {"precision": 0.28, "recall": 0.15, "ndcg": 0.32, "map": 0.09, "r_precision": 0.21}

{"tool": "eval_ranking", "arguments": {"rating_true": "<test_uri>", "rating_pred": "<sar_pred_uri>", "k": 10}}
// → {"precision": 0.33, "recall": 0.18, "ndcg": 0.38, "map": 0.11, "r_precision": 0.25}
```

Agent 向用户汇报：SAR 在此数据集上全面优于 NCF（Precision +15%，nDCG +16%）。

---

## 3. SASRec Amazon — 序列推荐模型

**Agent 场景**：用户说"用 SASRec 训练一个序列推荐器"。

### Step 1 — 加载 Amazon 评论数据

SASRec 脚本使用 `recommenders.datasets.amazon_reviews.get_review_data`，直接在脚本内部加载：

```bash
python skill/scripts/sasrec_amazon.py --size 100k --epochs 5 --model-out --state-root /app/state
```

### Step 2 — 训练输出

```json
{"status": "trained", "rows": 100000}
MODEL_HANDLE=00112233445566778899aabbccddeeff
```

### Step 3 — 后续推理

```python
from mcp_server.state import StateStore

store = StateStore("/app/state")
model = store.get_model("00112233445566778899aabbccddeeff")
# SASREC.predict(inputs) — inputs 为序列化的用户交互历史
```

**注意**：SASRec 需要 GPU 镜像。在 core 镜像中运行会因 `import recommenders.models.sasrec.model` 失败而触发延迟导入错误。

---

## 4. LightGBM Criteo — CTR 预估

**Agent 场景**：用户说"用 LightGBM 做 Criteo 点击率预估"。

### Step 1 — 加载 Criteo 数据

```json
// Agent 调用
{"tool": "load_criteo", "arguments": {"size": "sample", "cache_path": "/tmp/recommenders"}}

// Server 返回
{
  "uri": "file:///tmp/recommenders/d4e5f6a7b8c9.parquet",
  "rows": 100000,
  "schema": {"label": "int64", "I1": "int64", "C1": "object", "...": "..."}
}
```

### Step 2 — 训练 LightGBM

```bash
python skill/scripts/lightgbm_tinycriteo.py --size sample --ratio 0.75 --model-out --state-root /app/state
```

输出：

```json
{"status": "trained", "train_rows": 75000, "test_rows": 25000}
MODEL_HANDLE=1122334455667788aabbccddeeff0011
```

### Step 3 — 评估分类指标

```json
// Agent 将 LightGBM 预测结果传入 eval_classification
{
  "tool": "eval_classification",
  "arguments": {
    "rating_true": "<test_df_uri>",
    "rating_pred": "<lgbm_pred_df_uri>"
  }
}

// Server 返回
{"auc": 0.723, "logloss": 0.512}
```

---

## 5. TF-IDF COVID-19 — 内容推荐

**Agent 场景**：用户说"用 TF-IDF 做一个 COVID-19 论文推荐器"。

### Step 1 — 训练 TF-IDF 模型

```bash
python skill/scripts/tfidf_covid.py --top-k 5 --model-out --state-root /app/state
```

输出：

```json
{"status": "trained", "topk_shape": [1000, 5]}
MODEL_HANDLE=aabbccddeeff00112233445566778899
```

### Step 2 — 后续推理

```python
from mcp_server.state import StateStore

store = StateStore("/app/state")
model = store.get_model("aabbccddeeff00112233445566778899")
topk = model.recommend_top_k_items(new_papers_df, k=5)
```

### Step 3 — 超越准确率评估

```json
// Agent 调用 eval_beyond_accuracy 评估推荐的多样性和新颖度
{
  "tool": "eval_beyond_accuracy",
  "arguments": {
    "train_df": "<train_df_uri>",
    "reco_df": "<topk_df_uri>"
  }
}

// Server 返回
{
  "diversity": 0.85,
  "novelty": 0.42,
  "serendipity": 0.31,
  "catalog_coverage": 0.67,
  "distributional_coverage": 0.58
}
```

---

## Appendix: Handling missing dependencies / 附录：缺依赖处理

When a tool or script requires an extra that is not installed, the server returns a structured `MissingExtraError`:

```json
{
  "error": "Symbol 'recommenders.models.ncf.ncf_singlenode.NCF' requires extra 'gpu'. Install with: pip install 'recommenders-ai[gpu]' or pull recommenders-mcp:gpu image. No module named 'tensorflow'"
}
```

The agent should:

1. **Parse** the `extra` field from the error message.
2. **Instruct** the user to switch images or install the extra:
   - Docker: `docker pull recommenders-mcp:gpu` (or rebuild with `--build-arg COMPUTE=gpu`)
   - pip: `pip install 'recommenders-ai[gpu]'`
3. **Retry** the operation after the dependency is available.

### Image tier quick reference

| Capability | core | gpu |
|---|:---:|:---:|
| Data loading (Movielens/Criteo/MIND) | ✅ | ✅ |
| 4 splitters | ✅ | ✅ |
| 4 evaluation tools | ✅ | ✅ |
| get_top_k | ✅ | ✅ |
| SAR / RLRMC / TF-IDF / LightGBM / BPR | ✅ | ✅ |
| NCF / SASRec / Wide&Deep / deeprec / newsrec | ❌ | ✅ |
| Spark / ALS | ❌ | ❌ |
