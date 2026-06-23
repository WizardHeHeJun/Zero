# Zero

> 🚧 占位 README · 项目建设中（WIP）

多 Agent 协作系统。三层运行架构：

- **编排层** — LangGraph（Supervisor / Worker、StateGraph、Checkpointer）
- **记忆层** — Zep / Mem0 / Graphiti（长期记忆、知识图谱）
- **存储层** — Postgres / Neo4j / Redis

技术栈以 Python 为主、TypeScript 为辅。

## 架构图

见 [`diagrams/`](diagrams/) —— 多 Agent 协作系统 · 记忆架构分层（含情感链路）。

## 环境准备（conda）

Python 隔离环境用 conda 管理，环境名 `affective-expression`（Python 3.12，对齐 `pyproject.toml` 的 `requires-python`）。

```powershell
# 重建环境（跨平台，依赖口径对齐 pyproject.toml）
conda env create -f environment.yml
conda activate affective-expression

# 验证
pytest -q
```

- [`environment.yml`](environment.yml) — 跨平台可重建（loose 约束）。
- [`environment.lock.yml`](environment.lock.yml) — 本机 win-64 精确版本快照。
- 建立过程与命令速查见 [`notes/2026-06-24-conda-env-setup.md`](notes/2026-06-24-conda-env-setup.md)。

> 仓库同时支持 **uv**（`uv.lock` + `[tool.uv]`）：`uv sync` 亦可建好 `.venv`。两条路径依赖口径都以 `pyproject.toml` 为准。

## 状态

代码尚未开始实现（`src/` 待建立）。本仓库当前仅含架构设计产物与基础配置；后续按实现逐步补充。

---

*本 README 为占位，内容待项目推进后完善。*
