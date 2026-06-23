# Zero

> 多 Agent 协作系统。当前已落地**情感表达子系统**（affective-expression）：编排骨架 + 真网络化全通道脚手架 + 端到端集成。

三层运行架构：

- **编排层** — LangGraph（Supervisor / Worker、StateGraph、Checkpointer）
- **记忆层** — Zep / Mem0 / Graphiti（长期记忆、知识图谱）
- **存储层** — Postgres / Neo4j / Redis

技术栈以 Python 为主、TypeScript 为辅。

## 已实现：情感表达子系统

把"人的情感表达"建模为一条**贝叶斯流水线**，落成多 Agent 编排：

```text
Stimulus → Perception → Appraisal(OCC 先验) → Value(在线 TD / 精度) → AffectCore(主动推断·后验采样 e*)
                                                              → Regulation(掩饰) → Expression(双通路·4 通道)
```

- 6 个 Worker Agent（`src/agents/`）+ Supervisor（`src/orchestration/`）；节点契约 `(state) -> dict` 只返回增量。
- 记忆层 `src/memory/`（显式 scope、任务完成节流）；存储层 `src/storage/`（Checkpointer + 图谱占位，接口对齐 Postgres/Graphiti）。
- 数学内核：OCC 评价 → RPE/精度 → 高斯积融合 → 后验采样 → 双通路（真笑/假笑）× 4 通道（FACS AU / 文本标签 / 生理 / 韵律）。

### 真网络化（optional `ml` extra：torch/numpy/librosa/scipy）

占位解析函数可**逐通道**替换为可训练 torch 模型，`ExpressionAgent` 契约不变：

| 通道 | 模型 | 数据集 | 训练脚本 |
| --- | --- | --- | --- |
| 全通道（合成 bootstrap） | `ExpressionDecoder` | 合成（无需外部数据） | `scripts/train_expression.py` |
| 韵律 prosody | `ProsodyDecoder` | RAVDESS | `scripts/train_prosody.py` |
| 生理 physiology | `PhysiologyDecoder` | WESAD | `scripts/train_physiology.py` |
| 表情 FACS AU | `FacsDecoder` | AffectNet / DISFA | `scripts/train_facs.py` |
| 文本→(v,a) 输入侧 | `TextAffectRegressor` | EmoBank | `scripts/train_text_affect.py` |

- 数据集获取清单见 **[DATASETS.md](DATASETS.md)**。
- `CompositeChannelDecoder` 可叠加注入多通道真模型；经 `build_graph(..., expression_decoder=…)` / `runner.run(..., expression_decoder=…)` 接入管线（编排层经鸭子类型 `ChannelDecoder` 引用，**不依赖 torch**）。

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
> 真网络化需安装 `ml` extra：`pip install -e ".[ml]"`（或在 conda 环境内装 torch/numpy/librosa/scipy）。

## 跑测试 / 端到端 demo

```powershell
# 全量测试（torch 缺失时 ml 用例自动 importorskip 跳过，核心套件不依赖 torch）
pytest -q

# 端到端 demo：合成训练 ExpressionDecoder → 注入管线 → 跑刺激序列（无需外部数据）
python -m scripts.demo_pipeline
```

## 真实数据训练（可选）

放任一数据集到 `data/`（见 [DATASETS.md](DATASETS.md)），跑对应脚本：

```powershell
python -m scripts.train_prosody --root data/ravdess --epochs 300
# 权重存 artifacts/（已 gitignore），再注入 CompositeChannelDecoder 接管线
```

## 文档

- **[DATASETS.md](DATASETS.md)** — 真网络化所需数据集清单（含获取方式/许可）。
- **[PROGRESS.md](PROGRESS.md)** — 工程进度与成果记录。
- 知识层 `ai-docs/`（模块三件套 / catalog / pitfalls，本地 harness 资产，不入版本库）。

## 状态

- **情感表达子系统**：编排骨架 + 真网络化全通道脚手架 + 端到端集成已完成，测试全绿（`pytest` / `ruff` / `mypy`）。
- 记忆层、存储层为占位实现（接口对齐 Postgres / Graphiti），真实后端与更多 Worker 角色待接入。
