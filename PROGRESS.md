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

### 阶段 6 — Neo4j 长期记忆适配器 + Postgres saver 加固

把阶段 5 铺好的 env 后端工厂补上真图库适配器，并修掉 Postgres 接线隐患——代码就绪，待真机验证。

- **Neo4jGraphStore**（`src/storage/graph_store.py`）：裸 Cypher 实现 `GraphStore` 协议，保**时序失效**语义（新事实置同 (scope,key) 旧事实 `invalid_at`、非物理删除；`query_facts(at=)` 带时间语境）；`build_graph_store` 加 `neo4j` 分支，缺驱动告警回退 InMemory（与 checkpointer 同款）。
- **Postgres saver 加固**（`src/storage/checkpointer.py`）：`_postgres_saver` 改持显式长连接 + `autocommit/prepare_threshold=0/dict_row`，避开新版 langgraph `from_conn_string` 是 context manager、退出即关连接的坑；首次 `.setup()` 建表。
- **容器化对齐**：`docker-compose.yml` app 的 `ZERO_MEMORY_BACKEND` 切 `neo4j` + 连接 env（`.env.example` 需手动补 `ZERO_NEO4J_{URI,USER,PASSWORD}`，注意 `ZERO_MEMORY_BACKEND` 只能定义一次）。
- **测试**：`tests/test_neo4j_graph_store.py` 工厂回退 2 例（本机直跑）+ 时序语义 2 例（`importorskip` + 连接探测，无实例优雅 skip）；`ruff`/`mypy` 干净。
- **仍待真机验证**：本机无 Docker，需在服务器 `docker compose up` + 装 `db` extra，跑通 Postgres 跨重启恢复 + Neo4j 时序语义。

### 阶段 7 — 语言层 + affect↔language 双向收敛回路 + OpenAI adapter

把设想「情感是内核、语言由上下文+检索+情感生成、两者相互判断得出最终表现」落成一条**带终止上限的双向收敛回路**（默认关、零回归）。
- **回路接线**（`graph.py`）：`mood` 后插门控 `language` 节点；`route_after_mood`（扩展原 `route_after_affect_core`，加 language 优先分支）+ `route_after_language`（不一致且 `iter<language_max_iters` 则回 language，否则进 regulation/expression）两个纯路由函数，独立可单测。
- **LanguageAgent**（`src/agents/language.py`，async）：用 e*+上下文+检索（复用 `recalled_disposition`）生成语言并反推情感；回路重写前 `reconcile_affect` 把 e* 向语言情感拉拢——**双向互调**（情感也被语言微调，e* 不再纯固定内核）。可注入 `LanguageModel` 协议，默认占位 `_TemplateLanguageModel`（torch/API-free）。
- **OpenAILanguageModel**（`src/agents/language_openai.py`，optional `llm` extra）：通用 OpenAI 兼容接口（`base_url` 可指 OpenAI/vLLM/第三方网关），**两段式** = 生成回应 + 独立二次调用客观给文本打 VAD（使「相互判断」真实有效）；client 可注入、未注入延迟 import openai；编排层与默认路径不依赖 openai。
- **state/贯通**：`AffectState` 加 `language_*`（含 `language_enabled` 门控、`language_max_iters` 终止上限）；`expression` 把语言并入最终表现；`runner.run` 贯通开关 + 注入 `language_model`。
- **async 化**：`LanguageModel.generate`/`LanguageAgent.__call__`/占位/相关测试改 async（真 LLM 网络 I/O 不阻塞事件循环，与 supervisor async 节点同款）。
- 新测：`test_routing`（扩展加 language 路由）、`test_language_agent`、`test_language_loop`（端到端回路收敛/上限/双向微调）、`test_language_openai`（fake async client，不依赖 openai）。
- 验证：`pytest` **96 passed**（79 + language 13 + adapter 4）；`ruff`/`mypy` 干净。

### 阶段 8 — Graphiti 语义记忆深度集成（侧信道，默认关、零回归）

把待办里「引入 Graphiti（实体抽取/向量检索）」落成一条**与确定性 GraphStore 并存的语义记忆侧信道**——不替换基础后端，不把 LLM/网络塞进 affect 数学的确定性热路径。

- **存储层**（`src/storage/graph_store.py`，纯新增）：`SemanticStore` 协议（`@runtime_checkable`，全异步 `add_episode`/`search`）+ `GraphitiGraphStore`（包 `graphiti_core.Graphiti`，scope/key→`group_id`，构造不连接、首次读写一次性建索引，LLM/embedder 复用 `ZERO_OPENAI_*`+`ZERO_GRAPHITI_MODEL`，Neo4j 复用 `ZERO_NEO4J_*`）+ `build_semantic_store`（env `ZERO_SEMANTIC_BACKEND`，默认空→None、缺驱动告警回退）。**现有 GraphStore/三后端一字不动**。
- **记忆层**（`src/memory/client.py`，纯新增）：`MemoryClient(store, *, semantic=…)`；`write_episode`（富文本 episode→语义记忆，强制 scope、仅任务完成节点）+ `recall`（语义/向量检索，吃 query 文本；无后端返回 `[]`）。确定性 `write`/`query` 不变。
- **编排层深度集成落点**（全门控 + 能力检测）：`state.recalled_context`；Supervisor 任务完成时**额外**写自然语言情感事件 episode；MemoryRecall 语义召回 → `recalled_context`；**LanguageAgent 检索串并入 `recalled_context`**——Graphiti 的实体/关系召回由此真正影响语言生成。无语义后端时全链 no-op。
- **依赖/容器**：`graphiti` extra（`graphiti-core>=0.3`，自带 neo4j/openai）；`docker-compose.yml` app 加注释式语义后端 env（默认注释关）。
- **测试**：`tests/test_graphiti_semantic_store.py` —— 工厂回退（本机直跑）+ **FakeSemanticStore 确定性接线**（write_episode/recall 路由、显式 scope、`recalled_context` 填充、进语言检索、无后端零回归）+ 实机 smoke（importorskip + LLM/连接探测优雅 skip）。
- **验证**：`pytest` **106 passed**（96 + 10 新）；`ruff`/`mypy` 干净。
- **仍待真机验证**：本机无 Neo4j/Docker/LLM；服务器 `pip install -e ".[graphiti]"` + `ZERO_SEMANTIC_BACKEND=graphiti` + `ZERO_OPENAI_*`/`ZERO_NEO4J_*`，跑通 episode 抽实体/关系 → 语义召回 → 进语言层。

### 阶段 9 — Graphiti 本地无 Docker 验证路径（kuzu 嵌入式 + .env 自动加载）

阶段 8 的真机验证原本要 Docker+Neo4j+LLM（太重）；本阶段让它**在本机纯命令行就能验**（用户优先本地）。

- **图库 env 可选**（`src/storage/graph_store.py`）：`_build_graphiti` 按 `ZERO_GRAPHITI_DB` 选 `neo4j`（默认/生产，`uri,user,password`）/ `kuzu`（嵌入式、无服务进程，`graph_driver=KuzuDriver(db=ZERO_KUZU_PATH)`）。⚠ **kuzu upstream 已 deprecated**（会被 graphiti-core 移除），仅作本地 smoke 跳板，钉 `kuzu>=0.6`；持久/生产仍走 neo4j。
- **健壮性**：`_coerce_dt` 归一边的 `valid_at/invalid_at`（防 kuzu 已知 bug getzep/graphiti#893 的 `str>datetime` 崩溃）；`_graphiti_store` 改宽容 `except Exception` 回退——语义是可选侧信道，**绝不拖垮主管线**。
- **命令行验证脚本** `scripts/verify_graphiti_local.py`：同一 user 跑两次刺激（共享一个 MemoryClient/图库连接），看第 2 次 `recalled_context` 非空 = 写 episode→抽实体→语义召回→进语言层 闭环跑通；脚本可选 `_load_dotenv()` 自动加载根目录 `.env`（python-dotenv，未装静默跳过）。
- **配置就位**：`graphiti` extra 加 `kuzu` + `python-dotenv`；`.env.example` 补 `ZERO_GRAPHITI_DB`/`ZERO_KUZU_PATH`、`ZERO_SEMANTIC_BACKEND` 默认注释关、OpenAI 段合并为"语言层+Graphiti 共用一处"、加"每个 KEY 只出现一次"提醒。**库代码不依赖 dotenv**（仅便捷脚本加载 `.env`；正式/容器由真实环境注入）。
- **测试**：`_coerce_dt` 确定性单测 + kuzu 实机 smoke（importorskip graphiti_core+kuzu + LLM 探测，优雅 skip）。
- **验证**：`pytest` **107 passed**（+1）；`ruff`/`mypy` 干净。
- **仍待真机跑通**：本机无 LLM key；用户带 OpenAI 兼容 key 跑 `python -m scripts.verify_graphiti_local` 即可本地验证闭环（无 Docker）。

### 阶段 10 — 文本输入侧句向量升级（词袋→语义编码，跨域泛化实测）

阶段 2 三-5 的 `TextAffectRegressor` 用哈希词袋（无语义泛化、跨域即失效，预测幅度被压在 ±0.12）；本步换预训练句向量编码器，干净隔离出「语义表示 vs 词袋」一个变量（词袋版零回归、并存作基线）。

- **句向量模型**（`src/agents/models/text_affect_regressor_st.py`，纯新增）：`STTextAffectRegressor` = **冻结** `all-MiniLM-L6-v2`(384维) 句向量 → 同款 MLP 头，只训头。编码器**非 module 成员**（经 `encode_texts`+`lru_cache` 单例延迟 import sentence-transformers），故 `state_dict` 仅含 MLP 头。
- **loader/脚本**（纯新增）：`emobank_st.load_emobank_embeddings` 复用 `read_emobank_rows`（从 `emobank.py` 抽出的共用 CSV 解析/归一化源）+ 预计算句向量；`scripts/train_text_affect_st.py` 全批量训练。**词袋版零回归**（`load_emobank` 改为复用 `read_emobank_rows`，行为不变）。
- **实测对比**（EmoBank 全量 10062 句、同 MLP 头、CPU）：loss `0.0156 → 0.0056`（降 64%）；幅度 ±0.12 → ±0.6；"furious" arousal `+0.05 → +0.73`（学到「愤怒=高唤起」需语义的映射）；**跨域**——口语「omg…lit…best night」词袋错判负(−0.07)/句向量判对(+0.52)，商业体裁「revenue disappointing」词袋几无反应(−0.02)/句向量识别负(−0.33)。坐实文献：更广域靠**语义表示**而非堆数据量。
- **依赖/成本**：新 `nlp` extra（`sentence-transformers`），独立于 `ml`、默认词袋路径不引入；首次联网下 MiniLM 权重(~80MB)。编码器**冻结**（CPU 务实选择），端到端微调可再涨但需 GPU。
- **测试**：`tests/test_emobank_st.py`（importorskip torch+sentence_transformers 优雅跳过）—— 共用解析、ST 头 forward、句向量 loader 维度、predict 范围、训练 smoke。
- **验证**：`pytest` **113 passed**（107 + 6 新）；`ruff`/`mypy`(39 源) 干净。

## 成果与验证

- **测试**：`pytest` 113 passed, 4 skipped（含 ml 缺 torch 跳过 2 + Graphiti 实机 smoke neo4j/kuzu 各跳过 1；本机有 sentence-transformers 故句向量 6 测全跑）；`ruff check`/`ruff format`/`mypy`(39 源文件) 干净。
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
- PR #6（已合并）：本地真后端（SQLite 落盘 + env 后端工厂）+ 容器化脚手架 + MemoryRecall/Mood 两个 Worker。
- 分支 `feat/neo4j-graphstore-backend`：Neo4j 长期记忆适配器（裸 Cypher 保时序失效）+ Postgres saver 加固 + README/文档口径同步。
- 分支 `feat/local-backends-and-agents`（语言层）：`LanguageAgent` + affect↔language 双向收敛回路（`route_after_mood`/`route_after_language`，双向互调 + 终止上限）+ `OpenAILanguageModel`（OpenAI 兼容接口，生成 + 独立 VAD 反推，`llm` extra），全程默认关、零回归。
- 阶段 8（PR #10，已合并 main）：Graphiti 语义记忆深度集成——`SemanticStore` 协议 + `GraphitiGraphStore` + `MemoryClient.write_episode/recall`，与确定性 GraphStore 并存的侧信道；Supervisor 写富 episode、MemoryRecall 语义召回 → `recalled_context` → LanguageAgent 检索（`graphiti` extra），默认关、零回归。
- 分支 `feat/graphiti-kuzu-local-verify`（阶段 9）：Graphiti 本地无 Docker 验证路径——`ZERO_GRAPHITI_DB` 选 neo4j/kuzu（嵌入式）、`_coerce_dt` 防 kuzu datetime bug、`scripts/verify_graphiti_local.py` + `.env` 自动加载、`.env.example` 去重补 kuzu 变量；`graphiti` extra 加 `kuzu`/`python-dotenv`。
- 分支 `chore/temp-main-launcher`（阶段 10，本次）：文本输入侧句向量升级——`STTextAffectRegressor`（冻结 MiniLM + MLP 头）+ `emobank_st`/`train_text_affect_st` + `test_emobank_st`，词袋版零回归并存作基线；实测 loss 降 64% + 跨域口语/商业体裁判对；新 `nlp` extra（`sentence-transformers`）。

## 待办（需外部介入或独立轨道）

- **放数据跑真实训练**：**EmoBank 已实跑**（词袋版 loss 0.016；句向量升级版 loss 0.0056、跨域更稳，见阶段 10）；其余 EULA-free 集（RAVDESS 韵律 / WESAD 生理）仍待放 `data/` 跑对应 `train_*`（脚手架已就绪）。
- **接真实后端**：本地已上 SQLite 落盘 + env 后端工厂；**Neo4j GraphStore 适配器已实现**（`Neo4jGraphStore` 裸 Cypher 保时序失效语义、`build_graph_store` 加 `neo4j` 分支 + 缺驱动告警回退，compose 已切 neo4j 后端）；**Postgres saver 已加固**（持显式长连接 + `autocommit/prepare_threshold/dict_row`，避开新版 `from_conn_string` 是 context manager、退出即关连接的坑）。**待真机验证**：在有 Docker 的服务器 `docker compose up` + 装 `db` extra，跑通 Postgres 跨重启恢复运行态 + Neo4j 时序语义（本机无 Docker，集成用例 `importorskip` + 连接探测优雅跳过）。**Graphiti 已深度集成（阶段 8）**：作为与确定性 GraphStore 并存的语义记忆侧信道接入（`SemanticStore`/`GraphitiGraphStore`/`write_episode`/`recall`，富 episode → 语义召回 → 语言层检索），`graphiti` extra、`ZERO_SEMANTIC_BACKEND=graphiti` 门控、默认关。**本地验证路径已就绪（阶段 9）**：`ZERO_GRAPHITI_DB=kuzu`（嵌入式、无 Docker/无服务）+ OpenAI 兼容 key，跑 `python -m scripts.verify_graphiti_local` 即可在本机验证闭环；**待用户带 LLM key 实跑确认**（本机无 key）。
- **扩 Worker 角色**：已加 MemoryRecall / Mood；可继续按 `/new-agent` 增加。
