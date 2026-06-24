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
        → Language(语言生成·affect↔language 双向回路·gated) → Regulation(掩饰/重评·gated) → Expression(双通路·4 通道)
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
| 文本→(v,a)·句向量升级 | `STTextAffectRegressor`（冻结 MiniLM + MLP 头，语义泛化/跨域更稳） | EmoBank | `scripts/train_text_affect_st.py` |

- 数据集获取清单见 **[DATASETS.md](DATASETS.md)**。
- `CompositeChannelDecoder` 可叠加注入多通道真模型；经 `build_graph(..., expression_decoder=…)` / `runner.run(..., expression_decoder=…)` 接入管线（编排层经鸭子类型 `ChannelDecoder` 引用，**不依赖 torch**）。

### 时间深度 / 心境（v2 · A.7 · 默认关）

给情绪加「回不去的过去」：慢变 `mood`（运行态，进 Checkpointer，不入图谱）按双稳动力学 `m' = inertia·m + gain·tanh(k·m) + drive·e*`（`gain·k=1.0 > 1−inertia=0.4` → pitchfork 双稳）演化，并作为**第三个精度加权先验**回馈 `AffectCore`。于是情绪轨迹**历史依赖**：持续负面把心境推入负盆后，轻微正向也拉不出（滞后 / 反刍）。经 `mood_enabled=True` 开启，默认关闭、对 v1 **零回归**。设计见 [PRP/affective-expression-v2/design.md](PRP/affective-expression-v2/design.md)，理论见 [notes/2026-06-23-…](notes/2026-06-23-emotion-math-and-llm-expression.md) 的 A.7。

### 语言层 + affect↔language 双向回路（含 OpenAI adapter · 默认关）

把"情感是内核、语言由上下文+检索+情感生成、两者相互判断得出最终表现"落成一条**带终止上限的双向收敛回路**：

- `mood` 之后、`expression` 之前插入门控的 `language` 节点（`src/agents/language.py`）：用 e* + 上下文 + 检索信息生成语言并反推其情感；条件边 `route_after_language` 比较语言情感与内核 e*，不一致且未达 `language_max_iters` 上限则回路重写，重写前 `reconcile_affect` 把 e* 向语言情感拉拢——**双向互调**（情感也被语言微调，e* 不再纯固定内核）。
- **可注入语言模型**（鸭子类型 `LanguageModel` 协议，同 `ChannelDecoder`）：默认占位 `_TemplateLanguageModel`（torch/API-free）；真接入用 `OpenAILanguageModel`（`src/agents/language_openai.py`，optional `llm` extra）——通用 OpenAI 兼容接口（`base_url` 可指 OpenAI / 本地 vLLM / 第三方网关），**两段式**：① 按目标情感生成回应 ② 独立再调一次客观给文本打 VAD，使"相互判断"真实有效。
- `language_enabled` 门控、默认关、对前序**零回归**；`runner.run(..., language_enabled=True, language_model=…)` 贯通。env：`ZERO_OPENAI_BASE_URL` / `ZERO_OPENAI_API_KEY`（回退 `OPENAI_*`）。

### 文本输出情绪补足（词工程 / steering / 重评 / 评价条件化 · 默认关）

把"生成的语言"从 `e*→4 档离散词→套模板`，补成**有粒度、可控、可重评、可评测**的表达（生物学+数学文献见 [notes/2026-06-24-…](notes/2026-06-24-text-output-emotion.md)）：

- **情绪词典层** `src/agents/emotion_lexicon.py`（纯函数、torch/API-free）：`affect_label`（VA 极坐标 8 扇区×强度细粒度词，远超旧 4 档）+ `motivational_system`（Panksepp 动机色彩）+ `affect_logit_bias`（NRC-VAD 词典桥 / 加权解码 `Δlogit=β·⟨φ(w),e*⟩`）+ `intensity_envelope`（ECM 句内情绪衰减）。**不动 `text_label`**（旧 4 档仍作通道值），零回归。
- **重评优先**（Gross 过程模型）：`RegulationAgent` 经 `regulation_strategy` 选 `suppression`（默认）/`reappraisal`（改构念意义而非末端砍输出，更不负、唤醒更低）。
- **评价条件化**（CPM/EMA）：`appraisal_conditioning_enabled` 把 OCC 评价结构（非仅最终 (v,a)）并入语言生成；默认关、对未感知该参数的注入模型零回归。
- **VA steering 适配器** `src/agents/language_steering.py`（开放权重，`steer` extra）：e\*=(v,a) 作 steering 坐标，`steering_delta=α(V·w_V+A·w_A)` 加到 LM 隐状态；纯函数 steering 核 torch-free 可测，`SteerBackend` 协议可注入、默认延迟 transformers hook 后端。
- **情商探针回归** `tests/test_ei_probe.py`（EmoBench 式：场景→管线→效价方向/动机系统/粒度命中率）。

### 本地真后端 + 容器化（env 驱动，optional `db` extra）

占位后端升级为**本地真持久化**，并为后续 Docker 部署铺好 env 开关——默认全内存、零依赖可跑，设 env 即切落盘/真后端：

| 维度 | 后端 | env |
| --- | --- | --- |
| 长期记忆图谱（确定性 (scope,key) 失效） | `InMemory`（默认）/ `SqliteGraphStore`（落盘、时序失效）/ `Neo4jGraphStore`（裸 Cypher、gated） | `ZERO_MEMORY_BACKEND` · `ZERO_GRAPH_DB` · `ZERO_NEO4J_{URI,USER,PASSWORD}` |
| 语义记忆侧信道（富 episode + 语义召回） | 无（默认）/ `GraphitiGraphStore`（LLM 抽实体/关系 + 向量检索，gated，`graphiti` extra） | `ZERO_SEMANTIC_BACKEND` · `ZERO_GRAPHITI_MODEL` · `ZERO_OPENAI_*` · `ZERO_NEO4J_*` |
| 运行态 Checkpointer | `InMemory`（默认）/ SQLite / Postgres（gated） | `ZERO_CHECKPOINT_BACKEND` · `ZERO_CHECKPOINT_DB` · `ZERO_PG_DSN` |

- **容器化**：`Dockerfile` + `docker-compose.yml`（postgres + neo4j + app）+ `.dockerignore` + `.env.example`；真后端驱动在 `db` extra（`pip install -e ".[db]"`）。服务器上 `docker compose up` 即接 Postgres/Neo4j 验证。
- **记忆读闭环**：`MemoryRecallAgent` 在管线开头读 user 长期情绪倾向，偏置 `Appraisal` 先验（reward 不变，TD 通路不动）；`MoodAgent` 把 v2 心境的双稳更新独立成节点（A.7）。二者默认门控关闭。

### 语义记忆侧信道（默认关、零回归）

`SemanticStore` 协议（`src/storage/graph_store.py`）让语义记忆**与确定性 GraphStore 并存、可换实现**——不替换基础后端，不把 LLM/网络塞进 affect 数学的确定性热路径。两种后端按体量选：

| 后端 | `ZERO_SEMANTIC_BACKEND` | 实质 | 服务/依赖 |
| --- | --- | --- | --- |
| **SqliteVectorStore**（推荐/轻量） | `sqlite_vec` | SQLite 存 episode+embedding、余弦相似度 Top-K 召回 | 无图库/无服务/无 Docker，仅需 `openai` |
| **GraphitiGraphStore** | `graphiti` | LLM 抽实体/关系入知识图谱 + 语义检索 | 需图库 Neo4j + `.[graphiti]`（⚠ kuzu 后端 Graphiti 有 FTS bug，不可用） |

- **记忆层** `MemoryClient.write_episode`（富 episode→语义记忆）/ `recall`（语义检索）；确定性 `write`/`query` 不变。无语义后端时二者 no-op/返回空。
- **深度集成落点**：Supervisor 任务完成时额外写自然语言情感事件 episode；MemoryRecall 语义召回 → `recalled_context` → **LanguageAgent 检索串并入**，语义记忆由此真正影响语言生成。`recall_enabled` 门控。
- embedding 走 OpenAI 兼容接口（`ZERO_OPENAI_*` + `ZERO_GRAPHITI_EMBED_MODEL`），两后端共用。

**本地验证（无 Docker / 无服务，sqlite_vec）**——命令行即可跑通闭环：

```powershell
pip install -e ".[llm]"          # 仅需 openai（embedding）；sqlite_vec 不要图库
Copy-Item .env.example .env      # 在 .env 里：取消 ZERO_SEMANTIC_BACKEND=sqlite_vec 注释、填真 OPENAI_API_KEY、按需改 EMBED_MODEL
python -m scripts.verify_graphiti_local   # 同一 user 跑两次，看 recalled_context 非空 = 语义召回闭环跑通
```

- 脚本自动加载根目录 `.env`（python-dotenv，未装则退回 shell 导出）；env 见 `.env.example`。
- 想要实体/关系知识图谱时再上 `graphiti` 后端（Neo4j Desktop/服务器）——`SemanticStore` 协议同形、编排层无感切换。
- 库代码不依赖 dotenv，只有该便捷脚本加载 `.env`；正式运行/容器由真实环境注入（secrets 走 `.env`、不入库）。

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
│   │   ├── emotion_lexicon.py   #   情绪词典层：细粒度词 / Panksepp 动机 / VAD 词典桥 / ECM 时间包络
│   │   ├── language_steering.py #   SteeringLanguageModel：VA steering 适配器（开放权重·steer extra）
│   │   ├── models/              #   真网络化 torch 解码器（expression/prosody/physiology/facs/text/composite）
│   │   └── datasets/            #   DataLoader：synthetic / ravdess / wesad / emobank(+_st 句向量) / facs_csv
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

**三通道已在真实公开数据上实跑验证**：文本 EmoBank（词袋 loss 0.016 / 句向量 0.0056，跨域更稳）、韵律 RAVDESS（loss 0.026，pitch 随 arousal 单调上升）、生理 WESAD（loss 0.024，stress 心率/皮电最高的应激→自主神经激活）。

> **预训练权重**：上面四个真实数据训练权重可直接从 Release [`weights-v0.2`](https://github.com/WizardHeHeJun/Zero/releases/tag/weights-v0.2)（real-data trained）下载，放到 `artifacts/` 即用；`weights-v0.1` 为合成 demo 权重。

## 文档

- **[DATASETS.md](DATASETS.md)** — 真网络化所需数据集清单（含获取方式/许可）。
- **[PROGRESS.md](PROGRESS.md)** — 工程进度与成果记录。
- 知识层 `ai-docs/`（模块三件套 / catalog / pitfalls，本地 harness 资产，不入版本库）。

## 状态

- **情感表达子系统**：编排骨架 + 真网络化全通道脚手架（文本/韵律/生理三通道已在真实公开数据上实跑验证，权重见 Release `weights-v0.2`）+ 语言层 affect↔language 双向回路 + **文本输出情绪补足**（词工程细粒度词典 / VA steering / 重评 / 评价条件化，默认关、零回归）+ 端到端集成已完成，测试全绿（`pytest` 149 passed / `ruff` / `mypy`）。
- 存储层已上真后端适配器（长期记忆 SQLite 落盘 + Neo4j 裸 Cypher；运行态 SQLite/Postgres saver），env 选后端、`db` extra 装驱动——代码就绪，待在有 Docker 的服务器真机验证。
- **语义记忆侧信道**（`SemanticStore`/`write_episode`/`recall`，富 episode → 语义召回 → 语言层检索，`recall_enabled` 门控、默认关）：轻量 `SqliteVectorStore`（`sqlite_vec`，无图库/无服务）**已本地端到端验证闭环通过** ✅；`GraphitiGraphStore`（`graphiti`，实体/关系知识图谱）为需要图谱时的重型选项，走 Neo4j（⚠ kuzu 后端 Graphiti 有 FTS bug，不可用）。更多 Worker 角色按需接入。
