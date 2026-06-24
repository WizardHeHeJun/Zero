# 工程进度与成果记录

> 情感表达子系统（affective-expression）的建设记录。更新于 2026-06-24。

## 缘起

从一个跨学科问题出发——"一个人的情感表达在生物学和数学上是怎么进行的"——把答案落成本仓库目标领域的产物：一个**多 Agent 情感表达系统**。
- 生物学侧：神经环路（杏仁核/岛叶/PFC/VTA）、面神经双通路、自主神经系统。
- 数学侧：OCC 评价（理性先验）+ 强化学习 RPE（价值/精度）+ 主动推断/自由能（后验采样）三者统一为一条贝叶斯流水线。

## 建设阶段

### 阶段 0 — PRP 工作流产出骨架
走 `/refine-prd → /generate-prp → /validate-prp → /execute-prp`：
- 6 个 Worker Agent（perception/appraisal/value/affect_core/regulation/expression）+ Supervisor + LangGraph StateGraph。
- 三层架构：编排（`src/orchestration`,`src/agents`）→ 记忆（`src/memory`）→ 存储（`src/storage`），单向依赖。
- 数学内核 `affect_math.py`：OCC 先验 → TD 更新/精度 → 高斯积融合 → 后验采样 → 双通路·4 通道解码。
- 守红线：节点 `(state)->dict` 增量契约、记忆任务完成节流 + 显式 scope、运行态(Checkpointer)与长期记忆分离。

### 阶段 1 — code-review 整改
项目级宏观审查后：
- 两个 BLOCK 经实测裁定（白名单格式误报、agents↔orchestration 同层已被 rules 覆盖）。
- 落实：层依赖测试补 `agents 不 import memory`、测试改用公开 query API、补数学内核直接单测。
- 技术债：`trace` 改 `Annotated[list, operator.add]` reducer（消 O(n²)）、`MemoryClient` 依赖 `GraphStore` 协议、`session_id` 默认绑 thread_id 防串味。

### 阶段 2 — 真网络化（逐通道，optional `ml` extra）
占位解析函数逐通道替换为可训练 torch 模型，`ExpressionAgent` 契约不变：
- **三-1** 合成 bootstrap：`ExpressionDecoder`（(v,a)→11 维全通道），打通"数据→训练→注入→推理→回归"。
- **三-2** 韵律：`ProsodyDecoder` ← RAVDESS（librosa 提音高/能量/过零率）。
- **三-3** 生理：`PhysiologyDecoder` ← WESAD（scipy R 波检测算心率 + EDA/Temp）；`CompositeChannelDecoder` 泛化为多通道叠加注入。
- **三-4** 表情：`FacsDecoder` ← AffectNet/DISFA（CSV 标注）。
- **三-5** 输入侧：`TextAffectRegressor`（文本→(v,a)）← EmoBank（稳定哈希词袋）。

### 阶段 3 — 端到端集成
- `build_graph(..., expression_decoder=…)` / `runner.run(..., expression_decoder=…)` 注入真解码器（鸭子类型 `ChannelDecoder`，编排层不依赖 torch）。
- `scripts/demo_pipeline.py`：合成训练 → 注入 → 跑刺激序列，**真模型驱动表达流过完整管线**（无需外部数据）。

### 阶段 4 — v2：时间深度 / 心境（A.7 滞后）

给情绪加「回不去的过去」。理论依据 `notes/2026-06-23-…` 的 A.7（Gottman 双稳动力系统），设计见 `PRP/affective-expression-v2/design.md`。
- `affect_math.py`：`mood_step`（双稳松弛 `m'=inertia·m+gain·tanh(k·m)+drive·e*`，`gain·k=1.0>1−inertia=0.4` → pitchfork）+ `fuse_terms`（多项精度融合，2 项时 ≡ `gaussian_fuse`，已单测断言）+ `MOOD_*` 常量。**没动 `gaussian_fuse`**。
- `AffectState.mood`（运行态，进 Checkpointer 不入图谱）+ `mood_enabled`（默认 `False`）；`affect_core` 开启时把"已在的心境"并入为第三个精度加权先验、采样后 `mood_step` 更新；`runner` 贯通 `mood_enabled`、`mood` 进轨迹。
- **加法、默认关、零回归**：关闭时 `affect_core` 原样走 `gaussian_fuse`，v1 行为不变。
- 新测：`test_mood_dynamics`（双稳两盆 / **滞后捕获** / 有界 / `fuse_terms`≡`gaussian_fuse`）、`test_mood_pipeline`（**历史依赖**：连灌 6 次负面后，同一温和刺激在"有负面过去"的 thread 比全新 thread 更负；默认关不产 mood）。
- 验证：`pytest` **65 passed**（v1 的 59 + mood 的 6）；`ruff`/`mypy` 干净。

### 阶段 5 — 本地真后端 + 容器化 + 两个新 Worker

把占位后端升级为本地真持久化、为后续 Docker 部署铺好 env 驱动开关；记忆从「仅写」补上「读」闭环。
- **本地真后端**：`SqliteGraphStore`（stdlib sqlite3，时序失效、落盘可跨进程/重启留存）+ `build_graph_store`（env `ZERO_MEMORY_BACKEND` 选 memory/sqlite）；`build_checkpointer` 加 env `ZERO_CHECKPOINT_BACKEND`（memory/sqlite/postgres，缺驱动回退 InMemory）；`runner` 默认经工厂取后端。默认全 memory → 零回归。
- **容器化就绪**：`Dockerfile` + `docker-compose.yml`（postgres + neo4j + app）+ `.dockerignore` + `.env.example`；`db` extra（langgraph-checkpoint-sqlite/postgres、psycopg、neo4j）。本机无 Docker，作部署脚手架，服务器 `docker compose up` 即接真后端。
- **MemoryRecallAgent**（`src/orchestration/`，注入 client）：管线开头读 user 长期倾向 → `recalled_disposition` 偏置 `AppraisalAgent` 先验 valence（reward 不变，TD 通路不动）——**闭合记忆 读↔写**。`recall_enabled` 门控。
- **MoodAgent**（`src/agents/`）：把 v2 心境的 `mood_step` 更新从 `affect_core` 抽成独立节点（affect_core 仍**读**心境融合，MoodAgent **写**）。`mood_enabled` 门控。
- 图重连：`memory_recall → … → affect_core → mood →（条件边）regulation/expression`；两新节点默认 no-op。
- 验证：`pytest` **79 passed**（+9 新测：SqliteGraphStore 持久化、MoodAgent、MemoryRecall 含端到端闭环）；`ruff`/`mypy` 干净。

## 成果与验证

- **测试**：`pytest` 59 passed；`ruff check`/`ruff format`/`mypy`(33 源文件) 干净。
- **测试覆盖**：节点契约、条件边路由、闭环轨迹、双通路差异、在线 TD 收敛、记忆节流/scope、层依赖、数学内核边界、各通道 loader（合成 fixture 真实跑通 librosa/scipy）、端到端注入。
- **端到端 demo 实测**：对负向刺激产出 `e*≈(-0.45, 0.61)` → "angry"，FACS 以 AU04/AU15 主导、心率 ~96bpm，全部由训练模型经 6 节点管线生成。

## 关键设计

- **逐通道渐进真网络化**：`CompositeChannelDecoder` 在解析占位上只覆盖有真模型的通道，其余回退占位——可无破坏地一通道一通道上真数据。
- **torch 隔离**：核心编排/Agent 不依赖 torch；torch 只在 `models/`、`datasets/`、`scripts/` 与 ml 测试，后者用 `importorskip` 跳过。
- **数据零改造接入**：合成与真实 loader 同形 `(X, Y)`，换数据源即可，模型/训练循环复用。

## 仓库结构（新增产物）

```text
src/{orchestration,agents,memory,storage}/   核心系统
src/agents/{models,datasets}/                 真网络化：解码器 + DataLoader
scripts/train_*.py · demo_pipeline.py         训练脚本 + 端到端 demo
tests/                                        59 用例（核心 + ml，ml 可跳过）
DATASETS.md                                   数据集清单
```

## 版本管理

- PR #1（已合并）：编排骨架 + code-review 整改 + diagrams。
- PR #2（已合并）：conda 环境配置（`environment.yml`/lock）。
- PR #3（已合并）：真网络化 三-1~三-5 + 端到端集成。
- PR #4（已合并）：v2 时间深度 / 心境（A.7 滞后），默认关、零回归。
- 分支 `feat/local-backends-and-agents`（待 PR）：本地真后端（SQLite 落盘 + env 后端工厂）+ 容器化脚手架 + MemoryRecall/Mood 两个 Worker。

## 待办（需外部介入或独立轨道）

- **放数据跑真实训练**：任一 EULA-free 集（RAVDESS/WESAD/EmoBank）→ `data/` → 跑 `train_*`（脚手架已就绪）。
- **接真实后端**：本地已上 SQLite 落盘 + env 后端工厂（容器化就绪）；服务器侧 Postgres/Neo4j 经 `docker compose up` + `db` extra 接入并验证（Graphiti/Neo4j GraphStore 适配器待补）。
- **扩 Worker 角色**：已加 MemoryRecall / Mood；可继续按 `/new-agent` 增加。
