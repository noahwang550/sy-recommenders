# recommenders-ai

MCP server + Agent Skill wrapper for [Microsoft Recommenders](https://github.com/microsoft/recommenders) v1.2.1.

---

## English

### What is this?

`recommenders-ai` wraps the upstream [Recommenders](https://github.com/microsoft/recommenders) library behind two surfaces:

1. **MCP server** — 12 atomic tools (data loading, splitting, evaluation, top-k ranking) exposed over **stdio** or **HTTP**, packaged into two Docker images (`core` / `gpu`).
2. **Agent Skill** — runnable training scripts, playbooks, and code snippets aligned with the upstream `examples/00_quick_start/` notebooks.

Training produces **persistent model handles** via `state.py` (cloudpickle + parquet + meta.json, default TTL 24 h) so fitted models survive across conversations.

### Two image tiers (no Spark)

| Image | Build arg | Upstream extra | Contents |
|---|---|---|---|
| `recommenders-mcp:core` | `COMPUTE=core` | (none) | Data / split / evaluate / top-k + SAR / RLRMC / TF-IDF / LightGBM / BPR |
| `recommenders-mcp:gpu` | `COMPUTE=gpu` | `[gpu]` | core + TensorFlow + PyTorch: NCF / SASRec / Wide&Deep / EmbDotBias / deeprec / newsrec |

Spark/ALS is **intentionally excluded** — no JDK dependency, smaller images, simpler CI.

### Install

Three installation modes. Pick the lightest one that covers your workload.

#### 1. pip (three extras)

```bash
# CPU development (core tools + dev deps)
pip install -e ".[dev]"

# GPU models (NCF, SASRec, etc.)
pip install -e ".[gpu,dev]"

# Everything (brave)
pip install -e ".[all]"
```

> **Note**: `numpy<2` is pinned in `pyproject.toml` because recommenders v1.2.1 uses `np.NaN` (removed in NumPy 2.0). `pip install recommenders-ai` resolves this automatically.

#### 2. Docker pull / build

```bash
# Build the core (CPU) image
docker build --build-arg COMPUTE=core -t recommenders-mcp:core .

# Build the GPU image (requires nvidia-docker)
docker build --build-arg COMPUTE=gpu -t recommenders-mcp:gpu .
```

#### 3. docker-compose

```bash
export MCP_HTTP_TOKEN=change-me
docker-compose up
```

`docker-compose.yml` defines two services: `recommenders-mcp-stdio` (interactive child process) and `recommenders-mcp-http` (persistent port 8080 with Bearer auth). Both mount a shared `mcp_state` volume.

### `.mcp.json` configuration

```json
{
  "mcpServers": {
    "recommenders-mcp-stdio": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "MCP_TRANSPORT=stdio",
        "recommenders-mcp:core"
      ]
    },
    "recommenders-mcp-http": {
      "url": "http://localhost:8080",
      "headers": {
        "Authorization": "Bearer ${MCP_HTTP_TOKEN}"
      }
    }
  }
}
```

### Quick start

One sentence → SAR training + evaluation:

```bash
python skill/scripts/sar_movielens.py --size 100k --model-out
```

Output (SAR Movielens 100k baseline, TOL ±0.05):

```json
{
  "precision": 0.330753,
  "recall": 0.176385,
  "ndcg": 0.382461,
  "map": 0.110591
}
MODEL_HANDLE=aabbccddeeff00112233445566778899
```

To train SAR on **your own data** instead of Movielens:

```bash
python skill/scripts/sar_custom.py --data your_ratings.parquet --col-user user_id --col-item item_id --col-rating score --top-k 10 --model-out
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport mode: `stdio` or `http` |
| `MCP_HTTP_TOKEN` | *(required for http)* | Bearer token for HTTP auth; server refuses to start without it |
| `MCP_HTTP_PORT` | `8080` | HTTP listen port |
| `MCP_LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `STATE_TTL_SECONDS` | `86400` | TTL for persisted artifacts (24 h default) |
| `STATE_CLEANUP_INTERVAL` | `3600` | Minimum seconds between lazy cleanup sweeps |
| `MCP_FILE_ROOTS` | *(empty)* | OS-separated list of additional directories allowed for `file://` reads |

### Tests

```bash
# CPU unit tests (PR gate scope, ~42 tests)
pytest tests -m "not notebooks and not spark and not gpu" --disable-warnings

# CPU smoke tests (needs network for real Movielens 100k)
pytest tests -m "notebooks and not spark and not gpu" --disable-warnings

# GPU smoke tests
pytest tests -m "not notebooks and not spark and gpu" --disable-warnings

# Coverage report
pytest --cov=mcp_server --cov-report=term-missing
```

Tests are categorised by pytest markers defined in `pyproject.toml`: `notebooks`, `gpu`, `spark`, `experimental`. The wrapper maintains its own `tests/test_groups.yml` independent of upstream.

### Skill reference

See [`skill/SKILL.md`](skill/SKILL.md) for the capability matrix, playbook index, and artifact handle semantics.

### Documentation

- [`docs/tools_reference.md`](docs/tools_reference.md) — 12 MCP tool signatures with JSON examples and error codes.
- [`docs/usage_examples.md`](docs/usage_examples.md) — 6 agent conversation flows (SAR / NCF / SASRec / LightGBM / TF-IDF / SAR Custom Data).
- [`docs/CODEMAP.md`](docs/CODEMAP.md) — repository layout and module dependency graph.

### License

MIT. Based on [Microsoft Recommenders](https://github.com/microsoft/recommenders) (MIT, Linux Foundation AI & Data).

```
Copyright (c) Recommenders contributors.
```

### Contributing

1. Fork and branch from `main`.
2. `pip install -e ".[dev]"` and ensure `pytest tests -m "not notebooks and not spark and not gpu"` passes.
3. Format with `black .`.
4. Open a PR — all CI checks must be green.

---

## 中文

### 这是什么？

`recommenders-ai` 将上游 [Recommenders](https://github.com/microsoft/recommenders) 库封装在两个层面：

1. **MCP server** — 12 个原子工具（数据加载 / 划分 / 评估 / top-k 排序），通过 **stdio** 或 **HTTP** 暴露，打包成两档 Docker 镜像（`core` / `gpu`）。
2. **Agent Skill** — 可执行的训练脚本、playbook、代码片段，与上游 `examples/00_quick_start/` 笔记本对齐。

训练产物通过 `state.py` 持久化为 **model handle**（cloudpickle + parquet + meta.json，默认 TTL 24 小时），跨会话可用。

### 两档镜像（不含 Spark）

| 镜像 | 构建参数 | 上游 extra | 包含能力 |
|---|---|---|---|
| `recommenders-mcp:core` | `COMPUTE=core` | （无） | 数据 / 划分 / 评估 / top-k + SAR / RLRMC / TF-IDF / LightGBM / BPR |
| `recommenders-mcp:gpu` | `COMPUTE=gpu` | `[gpu]` | core + TensorFlow + PyTorch：NCF / SASRec / Wide&Deep / EmbDotBias / deeprec / newsrec |

Spark/ALS **不在封装范围内** — 省去 JDK 依赖、镜像更小、CI 更简单。

### 安装

三种安装方式，按需选最轻的。

#### 1. pip（三档 extras）

```bash
# CPU 开发（core 工具 + 开发依赖）
pip install -e ".[dev]"

# GPU 模型（NCF、SASRec 等）
pip install -e ".[gpu,dev]"

# 全部（谨慎）
pip install -e ".[all]"
```

> **注意**：`pyproject.toml` 已锁定 `numpy<2`，因为 recommenders v1.2.1 使用了 `np.NaN`（NumPy 2.0 已移除）。`pip install recommenders-ai` 会自动解析此约束。

#### 2. Docker 构建

```bash
# 构建 core（CPU）镜像
docker build --build-arg COMPUTE=core -t recommenders-mcp:core .

# 构建 GPU 镜像（需要 nvidia-docker）
docker build --build-arg COMPUTE=gpu -t recommenders-mcp:gpu .
```

#### 3. docker-compose

```bash
export MCP_HTTP_TOKEN=change-me
docker-compose up
```

`docker-compose.yml` 定义了两个服务：`recommenders-mcp-stdio`（交互式子进程）和 `recommenders-mcp-http`（常驻 8080 端口 + Bearer 鉴权），共享 `mcp_state` 数据卷。

### `.mcp.json` 配置

```json
{
  "mcpServers": {
    "recommenders-mcp-stdio": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "MCP_TRANSPORT=stdio",
        "recommenders-mcp:core"
      ]
    },
    "recommenders-mcp-http": {
      "url": "http://localhost:8080",
      "headers": {
        "Authorization": "Bearer ${MCP_HTTP_TOKEN}"
      }
    }
  }
}
```

### 快速开始

一句话 → SAR 训练 + 评估：

```bash
python skill/scripts/sar_movielens.py --size 100k --model-out
```

输出（SAR Movielens 100k 基准，容差 ±0.05）：

```json
{
  "precision": 0.330753,
  "recall": 0.176385,
  "ndcg": 0.382461,
  "map": 0.110591
}
MODEL_HANDLE=aabbccddeeff00112233445566778899
```

在**自有数据**上训练 SAR（而非 Movielens）：

```bash
python skill/scripts/sar_custom.py --data your_ratings.parquet --col-user user_id --col-item item_id --col-rating score --top-k 10 --model-out
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | 传输模式：`stdio` 或 `http` |
| `MCP_HTTP_TOKEN` | *(http 模式必填)* | HTTP Bearer 鉴权令牌；未设置时 server 拒绝启动 |
| `MCP_HTTP_PORT` | `8080` | HTTP 监听端口 |
| `MCP_LOG_LEVEL` | `INFO` | Python 日志级别（`DEBUG`、`INFO`、`WARNING`、`ERROR`） |
| `STATE_TTL_SECONDS` | `86400` | 持久化产物物的 TTL（默认 24 小时） |
| `STATE_CLEANUP_INTERVAL` | `3600` | 惰性清理的最小间隔秒数 |
| `MCP_FILE_ROOTS` | *(空)* | OS 分隔符分隔的额外目录列表，允许 `file://` 读取 |

### 测试

```bash
# CPU 单测（PR gate 范围，约 42 条）
pytest tests -m "not notebooks and not spark and not gpu" --disable-warnings

# CPU smoke 测试（需要网络下载真实 Movielens 100k）
pytest tests -m "notebooks and not spark and not gpu" --disable-warnings

# GPU smoke 测试
pytest tests -m "not notebooks and not spark and gpu" --disable-warnings

# 覆盖率报告
pytest --cov=mcp_server --cov-report=term-missing
```

测试按 pytest marker 分类，定义于 `pyproject.toml`：`notebooks`、`gpu`、`spark`、`experimental`。wrapper 维护独立的 `tests/test_groups.yml`，与上游完全隔离。

### Skill 参考

详见 [`skill/SKILL.md`](skill/SKILL.md)：能力矩阵、playbook 索引、训练产物 handle 语义。

### 文档

- [`docs/tools_reference.md`](docs/tools_reference.md) — 12 个 MCP 工具的签名、JSON 示例、错误码。
- [`docs/usage_examples.md`](docs/usage_examples.md) — 6 个 agent 对话流程（SAR / NCF / SASRec / LightGBM / TF-IDF / SAR 自有数据）。
- [`docs/CODEMAP.md`](docs/CODEMAP.md) — 仓库目录结构与模块依赖关系。

### 许可

MIT 许可。基于 [Microsoft Recommenders](https://github.com/microsoft/recommenders)（MIT，Linux Foundation AI & Data）。

```
Copyright (c) Recommenders contributors.
```

### 贡献

1. Fork 仓库并从 `main` 分支。
2. `pip install -e ".[dev]"` 并确保 `pytest tests -m "not notebooks and not spark and not gpu"` 通过。
3. 使用 `black .` 格式化。
4. 开 PR — 所有 CI 检查必须绿色。
