# Agent Platform Integration Guide

How to install and use **recommenders-ai** from any agent platform — Claude Code, Cursor, Continue, Cline, or a generic MCP/HTTP client.

> 本文档详细说明如何在各类 agent 平台上安装与使用 recommenders-ai：Claude Code、Cursor、Continue、Cline，以及通用 MCP/HTTP 客户端。

---

## 0. What you are wiring up / 你要接入的两层

`recommenders-ai` exposes **two independent surfaces**. You can use either or both:

| Surface | What it gives the agent | How the agent reaches it |
|---|---|---|
| **MCP server** (16 atomic tools) | `load_*`, `split_*`, `eval_*`, `get_top_k`, `recommend`, `list/describe/delete_handle` | **stdio** (MCP-native) or **HTTP** (custom REST, see §3.6) |
| **Agent Skill** (`skill/`) | 8 training scripts + 6 playbooks + snippets — domain knowledge for *when* to use which model | Bundled with the repo; load as a Skill in skill-aware platforms (§5) |

The **MCP server** is the runtime; the **Skill** is the playbook layer. Most agent platforms only need the MCP server. Skill-aware platforms (Claude Code) additionally benefit from loading `skill/`.

---

## 1. Prerequisites / 前置条件

| Option | Needs | Best for |
|---|---|---|
| **Docker** (recommended) | Docker 20+ | All platforms, no local Python, isolated, reproducible |
| **Local Python** | Python 3.11 (only) on host | Development / non-Docker hosts |

- **Python 3.11 is required** (`requires-python = ">=3.11,<3.12"`). 3.10/3.12 will not work.
- `numpy<2` is pinned — do not override it (recommenders v1.2.1 uses `np.NaN`, removed in NumPy 2.0).
- **GPU models** (NCF/SASRec/Wide&Deep/…) additionally need an NVIDIA runtime (`nvidia-container-toolkit` for Docker, or CUDA on host). CPU models (SAR/TF-IDF/LightGBM/RLRMC/BPR) need nothing special.

---

## 2. Step 1 — Get the server / 获取 server

### 2A. Docker image (recommended, all platforms)

```bash
git clone https://github.com/noahwang550/sy-recommenders.git
cd recommenders-ai

# CPU image (SAR / TF-IDF / LightGBM / RLRMC / BPR)
docker build --build-arg COMPUTE=core -t recommenders-mcp:core .

# GPU image (adds TensorFlow + PyTorch) — requires nvidia-docker
docker build --build-arg COMPUTE=gpu -t recommenders-mcp:gpu .
```

Verify it runs:

```bash
docker run --rm -e MCP_TRANSPORT=stdio recommenders-mcp:core recommenders-mcp --help
```

### 2B. pip install (no Docker)

```bash
git clone https://github.com/noahwang550/sy-recommenders.git
cd recommenders-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # CPU dev
# pip install -e ".[gpu,dev]"    # + GPU models (needs CUDA)
```

The console entry point is `recommenders-mcp` (= `python -m mcp_server.server`).

> **Windows + Git Bash note**: when invoking Docker from MSYS2/Bash, prefix commands with `MSYS_NO_PATHCONV=1` so `/app`-style paths are not mangled.

---

## 3. Step 2 — Connect your agent platform / 接入 agent 平台

> **Transport choice (read first).**
> - **stdio** is the **MCP-native** transport. Use it for Claude Code, Cursor, Continue, Cline — every standard MCP client.
> - **HTTP** here is a **custom lightweight REST API** (`GET /health`, `GET /tools`, `POST /invoke {tool, arguments}`), **not** the MCP streamable-HTTP/SSE protocol. It is meant for programmatic/curl-style access (proven in the demo loop). MCP clients that expect the standard MCP-over-HTTP protocol will **not** work against it — for those, use stdio.

### 3.1 Claude Code (Anthropic CLI / IDE)

Two scopes:

**Project-scoped** (committed with the repo) — create `.mcp.json` in the repo root (already shipped with this repo):

```json
{
  "mcpServers": {
    "recommenders-stdio": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "MCP_TRANSPORT=stdio", "recommenders-mcp:core"]
    }
  }
}
```

**User-scoped** (available in every project) — add the same block under `mcpServers` in `~/.claude.json`, or run:

```bash
claude mcp add recommenders-stdio -- docker run -i --rm -e MCP_TRANSPORT=stdio recommenders-mcp:core
```

**Local-pip variant** (no Docker):

```json
{
  "mcpServers": {
    "recommenders-stdio": {
      "command": "recommenders-mcp",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

Then in Claude Code: `/mcp` lists connected servers; the 16 tools appear as `mcp__recommenders-stdio__<tool>`.

### 3.2 Cursor

Create `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "recommenders-stdio": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "MCP_TRANSPORT=stdio", "recommenders-mcp:core"]
    }
  }
}
```

Restart Cursor → the tools are available in Composer/Chat. (Cursor also reads the project `.mcp.json` directly.)

### 3.3 Continue (VS Code / JetBrains)

Add to `~/.continue/config.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "recommenders-stdio": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "MCP_TRANSPORT=stdio", "recommenders-mcp:core"]
    }
  }
}
```

Reload the Continue window; invoke tools from chat with `@recommenders-stdio`.

### 3.4 Cline (VS Code extension)

Run the command **Cline: Open MCP Settings** (writes `cline_mcp_settings.json`), or paste:

```json
{
  "mcpServers": {
    "recommenders-stdio": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "MCP_TRANSPORT=stdio", "recommenders-mcp:core"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### 3.5 Any stdio MCP client (generic)

The server is a standard stdio MCP server. Spawn it as a child process with env `MCP_TRANSPORT=stdio`. Command matrix:

| Runtime | command | args |
|---|---|---|
| Docker | `docker` | `run -i --rm -e MCP_TRANSPORT=stdio recommenders-mcp:core` |
| Docker GPU | `docker` | `run -i --rm --gpus all -e MCP_TRANSPORT=stdio recommenders-mcp:gpu` |
| Local pip | `recommenders-mcp` | (none; set env `MCP_TRANSPORT=stdio`) |

State (model handles) persists in `/app/state` inside the container. To share handles across invocations, mount a host volume: add `-v recommenders-state:/app/state` (named) or `-v "$PWD/state:/app/state"` (bind).

### 3.6 Generic HTTP (custom REST, not MCP-over-HTTP)

Start the HTTP server (Docker):

```bash
docker run -d --rm -p 8080:8080 \
  -e MCP_TRANSPORT=http \
  -e MCP_HTTP_TOKEN=change-me \
  -v recommenders-state:/app/state \
  recommenders-mcp:core
```

Or `docker-compose up` (uses `MCP_HTTP_TOKEN` from env). Then call it with any HTTP client (Bearer auth, fail-closed):

```bash
# health
curl -s -H "Authorization: Bearer change-me" http://localhost:8080/health
# {"status":"ok"}

# list tools
curl -s -H "Authorization: Bearer change-me" http://localhost:8080/tools

# invoke a tool
curl -s -X POST http://localhost:8080/invoke \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"tool":"recommend","arguments":{"model_handle":"<32-hex>","user_data":"file:///app/users.parquet","top_k":10}}'
```

`user_data` accepts a JSON string (`orient=split`) or a `file://...` URI to a parquet/json file inside an allowed root (cwd / tempdir / `MCP_FILE_ROOTS`). Pickle is never accepted across the boundary.

---

## 4. Step 3 — Quick verify / 快速验证

After wiring stdio into your platform, ask the agent:

> "List the recommenders-mcp tools, then call `get_top_k` on a tiny DataFrame."

For HTTP, run the `curl /health` + `curl /tools` lines above. Then run the full **train→recommend loop** (the real production check) — see the MUJI example in [`docs/usage_examples.md`](usage_examples.md) and the v0.2.0 section of [`README.md`](../README.md):

```
1. Train   sar_custom.py --data your.parquet --col-rating score --model-out   → MODEL_HANDLE=<id>
2. Inspect mcp describe_handle(handle=<id>)                                   → {kind:model, size_bytes:...}
3. Score   mcp recommend(model_handle=<id>, user_data=users, top_k=10)        → {uri, rows, skipped_user_count, model_handle}
4. Eval    mcp eval_ranking(rating_true=test, rating_pred=recs, k=10)         → {precision, recall, ndcg, map}
5. Cleanup mcp delete_handle(handle=<id>)
```

---

## 5. Step 4 — Load the Agent Skill (optional, skill-aware platforms) / 加载 Skill

The `skill/` directory (`SKILL.md` + `playbooks/` + `scripts/` + `snippets/`) is the **domain-knowledge layer** — it tells the agent *which* model to pick for *which* problem (e.g. SAR for warm users, TF-IDF for cold items, LightGBM for CTR). It is independent of the MCP server.

### Claude Code (skills supported natively)

Copy `skill/` into a skills directory and ensure `SKILL.md` carries frontmatter:

```bash
# project-scoped
mkdir -p .claude/skills/recommenders-ai
cp -r skill/* .claude/skills/recommenders-ai/

# or user-scoped
mkdir -p ~/.claude/skills/recommenders-ai
cp -r skill/* ~/.claude/skills/recommenders-ai/
```

If `skill/SKILL.md` lacks YAML frontmatter, prepend:

```yaml
---
name: recommenders-ai
description: Build, evaluate, and deploy recommendation pipelines with Microsoft Recommenders v1.2.1. Use when training/scoring recommenders (SAR, TF-IDF, LightGBM, NCF, SASRec) or running the train→recommend loop over MCP.
---
```

Then `/skills` lists `recommenders-ai`; the agent loads the playbooks on demand.

### Other platforms

Cursor / Continue / Cline do not yet auto-load a skill bundle, but the artifacts are still usable as **reference docs the agent can read**: point the agent at `skill/SKILL.md` and `skill/playbooks/*.md` (e.g. "read skill/playbooks/06_recommend.md and run the loop"). The 8 scripts under `skill/scripts/` run standalone with the installed package.

---

## 6. Environment variables / 环境变量

| Variable | Default | Notes |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` (MCP-native) or `http` (custom REST) |
| `MCP_HTTP_TOKEN` | *(http required)* | Bearer token; server refuses to start if unset/empty |
| `MCP_HTTP_PORT` | `8080` | HTTP listen port |
| `MCP_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `STATE_ROOT` | `./state` | Where model handles persist |
| `STATE_TTL_SECONDS` | `86400` | Handle TTL (24 h) |
| `STATE_CLEANUP_INTERVAL` | `3600` | Lazy cleanup sweep interval |
| `MCP_FILE_ROOTS` | *(empty)* | Extra dirs allowed for `file://` reads (OS path-list) |
| `COMPUTE` | `core` | Set to `gpu` for the GPU image |

---

## 7. Troubleshooting / 排错

| Symptom | Fix |
|---|---|
| `docker: image recommenders-mcp:core not found` | Build it first — §2A. The `.mcp.json` references a built image, not a registry pull. |
| Server exits at startup in HTTP mode | `MCP_HTTP_TOKEN` is unset — set it (fail-closed by design). |
| `AttributeError: np.NaN was removed` | NumPy 2.x leaked in. Ensure `numpy<2`; with Docker, rebuild the image. |
| Claude Code HTTP MCP client fails to connect | The HTTP transport is a custom REST API, not MCP-over-HTTP. Use the **stdio** server entry instead (§3.1). |
| Windows/Git Bash: container can't see `/app` paths | Prefix Docker calls with `MSYS_NO_PATHCONV=1`. |
| `recommend` returns `skipped_user_count = N` | Those users were not in the training split — expected for cold users. Route them to `tfidf_custom.py` for content-based fallback. |
| Tool returns `{"error":..., "code":"state_not_found"}` (HTTP 410) | Handle expired (TTL 24 h) or was deleted — retrain. |
| `file://` read rejected as outside allowed roots | Add the dir to `MCP_FILE_ROOTS`, or move the file under cwd/tempdir. |

---

## 8. Platform compatibility matrix / 平台兼容性

| Platform | stdio (MCP-native) | HTTP (custom REST) | Skill bundle |
|---|---|---|---|
| Claude Code | ✅ `.mcp.json` | ⚠️ curl/script only | ✅ `.claude/skills/` |
| Cursor | ✅ `.cursor/mcp.json` | ⚠️ curl only | 📖 as reference docs |
| Continue | ✅ `config.json` | ⚠️ curl only | 📖 as reference docs |
| Cline | ✅ `cline_mcp_settings.json` | ⚠️ curl only | 📖 as reference docs |
| Generic MCP client | ✅ spawn stdio | ✅ `/invoke` REST | n/a |
| Custom agent / script | ✅ stdio | ✅ `/invoke` REST | 📖 as reference docs |

✅ = first-class support · ⚠️ = works via the custom REST surface (not MCP-over-HTTP) · 📖 = read the playbooks as docs.
