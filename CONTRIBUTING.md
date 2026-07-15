# Contributing to recommenders-ai

Thanks for your interest! This guide covers the local dev loop, the test
discipline, and the rules that keep this wrapper independent of the upstream
`recommenders` clone it lives inside.

> 贡献指南：本地开发流程、测试纪律，以及保持 wrapper 与上游 recommenders 克隆相互独立的核心规则。

---

## 1. Repo layout you must respect / 仓库边界

This repository is an **independent wrapper** around upstream
[Microsoft Recommenders](https://github.com/microsoft/recommenders) v1.2.1.
The `recommenders/` Python package is a *runtime dependency installed via pip*,
**not** source you edit here.

- Only modify files under `mcp_server/`, `skill/`, `tests/`, `docs/`, and the
  top-level config files.
- **Never** edit the upstream `recommenders/` package sources, even if this
  clone sits inside a checkout of that repo.
- Keep the two transports (`mcp_server/http_transport.py`, `mcp_server/server.py`)
  in sync: a new tool must be registered in `_register_all` **and** surface
  through the `_TOOL_REGISTRY` so both stdio and HTTP reach it.

---

## 2. Local setup / 本地环境

Python **3.11 only** (`>=3.11,<3.12`). There is no requirement for a system
Python if you use Docker.

```bash
git clone https://github.com/noahwang550/sy-recommenders.git
cd recommenders-ai
python -m venv .venv && source .venv/bin/activate    # Win: .venv\Scripts\activate
pip install -e ".[dev]"                                # CPU dev
# pip install -e ".[gpu,dev]"                           # + GPU (needs CUDA)
```

Docker (no local Python needed):

```bash
docker build --build-arg COMPUTE=core -t recommenders-mcp:core .
# Windows/Git Bash: prefix Docker calls with MSYS_NO_PATHCONV=1
```

Verify the numpy pin holds after every fresh install (recommenders v1.2.1 uses
`np.NaN`, removed in NumPy 2.0):

```bash
python -c "import numpy; print(numpy.__version__)"     # expect 1.26.x
```

---

## 3. Test discipline / 测试纪律 (TDD)

This project follows **write-tests-first (RED → GREEN → REFACTOR)**.

1. Write a failing test under `tests/` that captures the behavior.
2. Run it and confirm it fails for the right reason.
3. Implement the minimal change to pass.
4. Run the full CPU suite to confirm no regression.
5. Add the new test path to `tests/test_groups.yml` (the group that runs in CI)
   — **a test not listed there will not execute in CI.**

### Run tests

```bash
# CPU unit tests (PR-gate scope)
pytest tests -m "not notebooks and not spark and not gpu" --disable-warnings

# CPU smoke tests (need network for real datasets)
pytest tests -m "notebooks and not spark and not gpu" --disable-warnings

# Coverage
pytest --cov=mcp_server --cov-report=term-missing
```

Markers (defined in `pyproject.toml`): `notebooks`, `gpu`, `spark`,
`experimental`. Tag new tests with the right marker so they land in the right
CI group.

### Assertions

- AAA structure (Arrange / Act / Assert).
- `assert computation == value`; use `pytest.approx` for floats; `is` only for
  singletons (`None`).
- Prefer many small tests over one large one; share data via `@pytest.fixture`.

---

## 4. Code style / 代码风格

- PEP 8 + **black** (line length 100, `target-version = py311`).
- Type hints on signatures; annotate `model` / `server` params as `Any` when
  duck-typed.
- Functions < 50 lines, files < 800 lines, nesting ≤ 4 levels.
- Immutable patterns; explicit error handling; no silent swallows.
- No hardcoded secrets; validate at boundaries.

```bash
black .          # format
black --check . # verify
```

---

## 5. Before opening a PR / 提交 PR 前

- [ ] `pytest tests -m "not notebooks and not spark and not gpu"` passes.
- [ ] `black --check .` is clean.
- [ ] `numpy<2` still holds in your env.
- [ ] New tests are registered in `tests/test_groups.yml`.
- [ ] If a tool changed, `docs/tools_reference.md` and `skill/SKILL.md` track it.
- [ ] No edits to upstream `recommenders/` sources.
- [ ] No secrets, `.parquet` fixtures, or `state/` artifacts committed
  (they are gitignored).

Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`perf:`, `ci:`, `chore:`).

---

## 6. Architecture & design context / 架构参考

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — invariants, tool domains,
  the state-store lifecycle, and the typed HTTP error envelope.
- [`docs/CODEMAP.md`](docs/CODEMAP.md) — module dependency graph.
- [`docs/AGENT_INTEGRATION.md`](docs/AGENT_INTEGRATION.md) — wiring the server
  into agent platforms.
- [`docs/tools_reference.md`](docs/tools_reference.md) — the 16 tool signatures.

The core invariants: tools are **stateless atoms**; model objects never cross
the MCP boundary (only handle id strings); HTTP auth is **fail-closed**;
`recommend` accepts only a `model_handle: str` and retrieves the model
server-side.
