# Zero

> 多 Agent 协作系统。当前已落地**情感表达子系统**（affective-expression）：编排骨架 + 真网络化全通道脚手架 + 语言层 affect↔language 双向回路 + 端到端集成。

三层运行架构：

- **编排层** — LangGraph（Supervisor / Worker、StateGraph、Checkpointer）
- **记忆层** — Zep / Mem0 / Graphiti（长期记忆、知识图谱）
- **存储层** — Postgres / Neo4j / Redis

技术栈以 Python 为主、TypeScript 为辅。

## 已实现：情感表达子系统

把"人的情感表达"建模为一条**贝叶斯流水线**，落成多 Agent 编排：

```text
Stimulus → MemoryRecall(读长期倾向·gated) → Perception → Appraisal(OCC 先验·可回灌偏置) → Value(在线 TD/精度)
        → AffectCore(主动推断·后验采样 e*) → Mood(心境双稳·gated A.7)
        → Language(语言生成·affect↔language 双向回路·gated) → Regulation(掩饰) → Expression(双通路·4 通道)
```

- 8 个 Worker（`src/agents/`）+ MemoryRecall（`src/orchestration/`）+ Supervisor；节点契约 `(state) -> dict` 只返回增量。`MemoryRecall`/`Mood`/`Language` 由 `recall_enabled`/`mood_enabled`/`language_enabled` 门控，默认 no-op、零回归。
- 记忆层 `src/memory/`（显式 scope、任务完成节流，**读↔写闭环**：Supervisor 写长期倾向、MemoryRecall 读回偏置 appraisal）；存储层 `src/storage/`（最底层；运行态 Checkpointer + 长期记忆图谱，env 选后端：运行态 InMemory/SQLite/Postgres、图谱 InMemory/SQLite/Neo4j，接口对齐 Graphiti）。
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

### 时间深度 / 心境（v2 · A.7 · 默认关）

给情绪加「回不去的过去」：慢变 `mood`（运行态，进 Checkpointer，不入图谱）按双稳动力学 `m' = inertia·m + gain·tanh(k·m) + drive·e*`（`gain·k=1.0 > 1−inertia=0.4` → pitchfork 双稳）演化，并作为**第三个精度加权先验**回馈 `AffectCore`。于是情绪轨迹**历史依赖**：持续负面把心境推入负盆后，轻微正向也拉不出（滞后 / 反刍）。经 `mood_enabled=True` 开启，默认关闭、对 v1 **零回归**。设计见 [PRP/affective-expression-v2/design.md](PRP/affective-expression-v2/design.md)，理论见 [notes/2026-06-23-…](notes/2026-06-23-emotion-math-and-llm-expression.md) 的 A.7。

### 语言层 + affect↔language 双向回路（含 OpenAI adapter · 默认关）

把"情感是内核、语言由上下文+检索+情感生成、两者相互判断得出最终表现"落成一条**带终止上限的双向收敛回路**：

- `mood` 之后、`expression` 之前插入门控的 `language` 节点（`src/agents/language.py`）：用 e* + 上下文 + 检索信息生成语言并反推其情感；条件边 `route_after_language` 比较语言情感与内核 e*，不一致且未达 `language_max_iters` 上限则回路重写，重写前 `reconcile_affect` 把 e* 向语言情感拉拢——**双向互调**（情感也被语言微调，e* 不再纯固定内核）。
- **可注入语言模型**（鸭子类型 `LanguageModel` 协议，同 `ChannelDecoder`）：默认占位 `_TemplateLanguageModel`（torch/API-free）；真接入用 `OpenAILanguageModel`（`src/agents/language_openai.py`，optional `llm` extra）——通用 OpenAI 兼容接口（`base_url` 可指 OpenAI / 本地 vLLM / 第三方网关），**两段式**：① 按目标情感生成回应 ② 独立再调一次客观给文本打 VAD，使"相互判断"真实有效。
- `language_enabled` 门控、默认关、对前序**零回归**；`runner.run(..., language_enabled=True, language_model=…)` 贯通。env：`ZERO_OPENAI_BASE_URL` / `ZERO_OPENAI_API_KEY`（回退 `OPENAI_*`）。

### 本地真后端 + 容器化（env 驱动，optional `db` extra）

占位后端升级为**本地真持久化**，并为后续 Docker 部署铺好 env 开关——默认全内存、零依赖可跑，设 env 即切落盘/真后端：

| 维度 | 后端 | env |
| --- | --- | --- |
| 长期记忆图谱（确定性 (scope,key) 失效） | `InMemory`（默认）/ `SqliteGraphStore`（落盘、时序失效）/ `Neo4jGraphStore`（裸 Cypher、gated） | `ZERO_MEMORY_BACKEND` · `ZERO_GRAPH_DB` · `ZERO_NEO4J_{URI,USER,PASSWORD}` |
| 语义记忆侧信道（富 episode + 语义召回） | 无（默认）/ `GraphitiGraphStore`（LLM 抽实体/关系 + 向量检索，gated，`graphiti` extra） | `ZERO_SEMANTIC_BACKEND` · `ZERO_GRAPHITI_MODEL` · `ZERO_OPENAI_*` · `ZERO_NEO4J_*` |
| 运行态 Checkpointer | `InMemory`（默认）/ SQLite / Postgres（gated） | `ZERO_CHECKPOINT_BACKEND` · `ZERO_CHECKPOINT_DB` · `ZERO_PG_DSN` |

- **容器化**：`Dockerfile` + `docker-compose.yml`（postgres + neo4j + app）+ `.dockerignore` + `.env.example`；真后端驱动在 `db` extra（`pip install -e ".[db]"`）。服务器上 `docker compose up` 即接 Postgres/Neo4j 验证。
- **记忆读闭环**：`MemoryRecallAgent` 在管线开头读 user 长期情绪倾向，偏置 `Appraisal` 先验（reward 不变，TD 通路不动）；`MoodAgent` 把 v2 心境的双稳更新独立成节点（A.7）。二者默认门控关闭。

### 语义记忆深度集成（Graphiti · 侧信道 · 默认关）

把 Graphiti 作为**与确定性 GraphStore 并存的语义记忆侧信道**接入——不替换基础后端，不把 LLM/网络塞进 affect 数学的确定性热路径：

- **存储层** `SemanticStore` 协议 + `GraphitiGraphStore`（`src/storage/graph_store.py`）：LLM 抽实体/关系入图 + 语义/向量检索，scope/key→Graphiti `group_id`；构造不连接、首次读写一次性建索引；LLM/embedder 复用 `ZERO_OPENAI_*` + `ZERO_GRAPHITI_MODEL`，Neo4j 复用 `ZERO_NEO4J_*`。
- **记忆层** `MemoryClient.write_episode`（富文本 episode→语义记忆）/ `recall`（语义检索）；确定性 `write`/`query` 不变。无语义后端时二者 no-op/返回空（零回归）。
- **深度集成落点**：Supervisor 任务完成时额外写自然语言情感事件 episode → Graphiti 抽实体/关系；MemoryRecall 语义召回 → `recalled_context` → **LanguageAgent 检索串并入**，语义图谱由此真正影响语言生成。全链 `ZERO_SEMANTIC_BACKEND=graphiti` + `recall_enabled` 门控，默认关。
- 装 `graphiti` extra：`pip install -e ".[graphiti]"`。**图库可选**：`ZERO_GRAPHITI_DB=neo4j`（默认，持久/生产）或 `kuzu`（嵌入式、本地无服务/无 Docker，⚠ upstream 已 deprecated，仅作本地 smoke）。

**本地验证（无 Docker）**：装 `.[graphiti]` + 配 `ZERO_SEMANTIC_BACKEND=graphiti` · `ZERO_GRAPHITI_DB=kuzu` · `ZERO_OPENAI_*` · `ZERO_GRAPHITI_MODEL`，跑：

```powershell
python -m scripts.verify_graphiti_local   # 同一 user 跑两次，看 recalled_context 非空 = 语义召回闭环跑通
```

持久/生产验证仍走 Neo4j（Desktop 或服务器）。

## 架构图

见 [`diagrams/`](diagrams/) —— 多 Agent 协作系统 · 记忆架构分层（含情感链路）。

## 项目结构

```text
Zero/
├── src/                         # 核心系统（三层架构，依赖单向：编排 → 记忆 → 存储）
│   ├── orchestration/           # 编排层：StateGraph 装配 + 运行入口
│   │   ├── graph.py             #   build_graph：10 节点装配 + 条件边路由（含 language 双向回路）
│   │   ├── state.py             #   AffectState / Stimulus（pydantic 结构化 state）
│   │   ├── supervisor.py        #   SupervisorAgent：协调 + 任务完成节流写记忆
│   │   ├── memory_recall.py     #   MemoryRecallAgent：读 user 长期倾向回灌（记忆读闭环）
│   │   └── runner.py            #   run()：跑刺激序列、收集 (v,a) 轨迹
│   ├── agents/                  # 编排层·各 Worker（节点契约 (state) -> dict 只回增量）
│   │   ├── affect_math.py       #   纯数学内核：OCC / TD / 精度 / 高斯融合 / mood_step / 语言距离·互调
│   │   ├── perception.py · appraisal.py · value.py
│   │   ├── affect_core.py       #   主动推断·后验采样 e*（随机性来源）
│   │   ├── mood.py              #   MoodAgent：慢变心境双稳更新（A.7 滞后）
│   │   ├── regulation.py · expression.py   # 掩饰 + 双通路·4 通道输出
│   │   ├── language.py          #   LanguageAgent：语言生成 + affect↔language 双向回路（gated）
│   │   ├── language_openai.py   #   OpenAILanguageModel：OpenAI 兼容接口 adapter（生成 + 独立 VAD 反推）
│   │   ├── models/              #   真网络化 torch 解码器（expression/prosody/physiology/facs/text/composite）
│   │   └── datasets/            #   DataLoader：synthetic / ravdess / wesad / emobank / facs_csv
│   ├── memory/                  # 记忆层：读写 API（显式 scope、任务完成节流）
│   │   └── client.py · types.py
│   └── storage/                 # 存储层（最底层）：运行态 + 长期记忆，env 选后端
│       ├── checkpointer.py      #   build_checkpointer：InMemory / SQLite / Postgres（gated）
│       └── graph_store.py       #   InMemory / Sqlite / Neo4jGraphStore（裸 Cypher）+ build_graph_store
├── tests/                       # 核心 + ml + 真后端用例（ml 缺 torch / Neo4j 缺实例时自动 importorskip / 优雅跳过）
├── scripts/                     # 训练脚本 train_*.py + 端到端 demo_pipeline.py
├── Dockerfile · docker-compose.yml · .dockerignore · .env.example   # 容器化部署
├── pyproject.toml               # 依赖（core / ml / db / llm extra）+ ruff / mypy / pytest 配置
├── environment.yml · environment.lock.yml · uv.lock                 # conda / uv 环境
├── README.md · PROGRESS.md · DATASETS.md                            # 总览 / 工程记录 / 数据集清单
├── notes/                       # 研究笔记（情感数学 / LLM 表达，含 A.7 推导）
└── diagrams/                    # 架构设计图
```

> **本地 harness / 知识层**（gitignore，不入版本库）：`.claude/`（rules · skills · hooks · commands · agents）· `ai-docs/`（模块三件套 · catalog · pitfalls）· `ai-shared/` · `evals/` · `PRP/`（PRP 工作区）· `artifacts/` · `data/`（训练权重 / 数据集）。

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
> 真语言层（OpenAI 兼容接口）需 `llm` extra：`pip install -e ".[llm]"`，并配 `ZERO_OPENAI_BASE_URL`/`ZERO_OPENAI_API_KEY`（回退 `OPENAI_*`）。

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

- **情感表达子系统**：编排骨架 + 真网络化全通道脚手架 + 语言层 affect↔language 双向回路 + 端到端集成已完成，测试全绿（`pytest` 106 / `ruff` / `mypy`）。
- 存储层已上真后端适配器（长期记忆 SQLite 落盘 + Neo4j 裸 Cypher；运行态 SQLite/Postgres saver），env 选后端、`db` extra 装驱动——代码就绪，待在有 Docker 的服务器真机验证。
- **Graphiti 语义记忆深度集成**（`SemanticStore`/`GraphitiGraphStore`/`write_episode`/`recall` 侧信道，富 episode → 语义召回 → 语言层检索），`graphiti` extra、`ZERO_SEMANTIC_BACKEND` 门控、默认关——代码就绪，待真机验证（需 Neo4j + LLM）。测试全绿（`pytest` 106 / `ruff` / `mypy`）。更多 Worker 角色按需接入。
