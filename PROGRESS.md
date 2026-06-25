# 工程进度与成果记录

> **AI 仿生人**（情感引擎 ⊗ LLM，拟人情感交流）的建设记录；当前主体为情感表达子系统（affective-expression）。更新于 2026-06-25。前瞻路线图见 [notes/2026-06-25-roadmap-bionic-human.md](notes/2026-06-25-roadmap-bionic-human.md)。

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

### 阶段 11 — 真实数据三通道全部实跑（3/3）+ 权重发布（weights-v0.2）+ 文档同步

把阶段 2 真网络化脚手架真正喂上真实公开数据：文本/韵律/生理三通道全部在真实数据上训练验证，权重发布到 Release，README/知识层同步。**RAVDESS/WESAD 为纯运行现成脚手架（无代码改动）；文本句向量升级的代码见阶段 10。**

- **文本 EmoBank**（输入侧）：词袋 `TextAffectRegressor` loss 0.016；句向量 `STTextAffectRegressor`（阶段 10）loss 0.0056、跨域口语/商业体裁判对——坐实「语义表示 > 词袋」。
- **韵律 RAVDESS**（Zenodo 免登录、全量 1440 条）：`ProsodyDecoder` loss 0.126→0.026；推理验证 **pitch 随 arousal 单调上升**的真实声学映射（F0 主由唤起度驱动，valence 次要）。
- **生理 WESAD**（uni-siegen sciebo 2.25GB、15 受试者切 1474 个 30s 窗）：`PhysiologyDecoder` loss 0.053→0.024；推理验证 **stress 心率 92.7bpm·皮电最高 vs meditation 67.5** 的应激→自主神经（交感/副交感）激活。⚠ 数据获取坑：UCI 只给指针、其 sciebo 链接 `pYjSgfOVs6Ntahr` 已失效，改用经典 `HGdUkoNlW1Ub0Gx`。
- **弱区分项**（韵律 speech_rate/energy、生理体温）属 loader 特征代理的归一化尺度问题，非模型问题。
- **权重发布**：GitHub Release `weights-v0.2`（real-data trained，4 个真实权重）；`weights-v0.1` 留作合成 demo。`artifacts/` 仍 gitignore，权重一律走 Release。
- **文档同步**：README（真网络化表加句向量版 + 三通道实跑 + weights 下载）、ai-docs agents guide「真网络化」节 + catalog 里程碑（ai-docs 为本地 gitignore 知识层）。

### 阶段 12 — 文本输出侧情绪补足（生物学 + 数学文献落地，默认关、零回归）

承接 `notes/2026-06-24-text-output-emotion.md` 的文献调研（生物学：躯体标记/Panksepp 七系统/语言产出脑回路+神经调质/Gross 调节；数学：VAD 词典桥、VA 子空间环状 steering、可控生成四代、评价驱动 NLG、情商评测），把「文本输出」从 `e*→4 档离散词→套模板」补成有粒度、可控、可重评、可评测的表达。**全部加法、协议注入、默认关、零回归。**

- **情绪词典层**（`src/agents/emotion_lexicon.py`，纯新增、torch/API-free）：`affect_label`（VA 极坐标 8 扇区 × 强度分级，粒度远超旧 4 档）、`motivational_system`（Panksepp 动机色彩 seeking/care/rage/panic_grief）、`SEED_VAD_LEXICON`+`affect_logit_bias`（NRC-VAD 词典桥/加权解码 `Δlogit=β·⟨φ(w),e*⟩`）+`suggest_affect_words`+`appraise_text`（词典法反推）、`intensity_envelope`（ECM 句内 sigmoid 情绪衰减）。**不动 `text_label`**（旧 4 档仍作通道值）。
- **评价向量条件化**（CPM/EMA/APTNESS）：`appraisal_conditioning_enabled` 门控，`LanguageAgent` 把 OCC 评价结构（非仅最终 (v,a)）经 `_appraisal_summary` 并入生成；`LanguageModel.generate` 加 `appraisal=""` 参数（**仅开启且有摘要时才传** → 对未感知该参数的注入模型零回归），`_TemplateLanguageModel`/`OpenAILanguageModel` 同步支持。
- **重评优先**（Gross 过程模型）：`affect_math.reappraise`（早期干预：负效价按积极锚重新解释上抬 + 唤醒平复，改体验）；`RegulationAgent` 按 `regulation_strategy` 选 `suppression`（默认，仍 (0.3v,0.5a)）/`reappraisal`。实测重评比抑制「更不负、唤醒更低」。
- **VA steering 适配器**（`src/agents/language_steering.py`，开放权重，`steer` extra）：纯函数 steering 核（`axis_from_contrast` eq.1、`l2_normalize`、`orthogonalize` Gram-Schmidt、`steering_delta=α(V·w_V+A·w_A)`，torch-free 可单测）+ `SteerBackend` 协议（注入缝）+ 默认 `_TransformersSteerBackend`（延迟 import torch/transformers、前向 hook 加 delta 到目标层）。e\*=(v,a) 即天然 steering 坐标。
- **OpenAI 词典提示**：`OpenAILanguageModel(use_lexicon=…)` 默认关；开启把 affect-congruent 建议词注入 compose 提示，二段式 VAD 反推充当 reranker。
- **EmoBench 式情商探针回归**（`tests/test_ei_probe.py`）：curated 情绪场景 → 端到端管线 → `affect_label`/`motivational_system` 命中率（效价方向 + 动机系统 + 粒度多样性），文档化如何接真 EmoBench。
- **测试**：`test_emotion_lexicon`(15)、`test_language_steering`(纯数学+fake backend+importorskip smoke)、`test_regulation`(策略)、EI 探针、`test_affect_math`/`test_language_agent` 增测。
- **验证**：`pytest` **149 passed, 5 skipped**（+32；新增 steering smoke skip 1）；`ruff check`/`ruff format`/`mypy`(41 源) 干净。
- **真 LLM 端到端已验证**（`main.py --llm`，OpenAI 兼容代理 + qwen-flash）：四情绪场景（喜/怒×强弱）生成语言情绪对路、独立反推 VAD 与内核 e* 同象限（狂喜↔狂喜、暴怒↔暴怒）、双向回路迭代 2–3 次将一致性收敛到 τ=0.15 以下（0.12–0.14）。模型经 `.env` 的 `ZERO_OPENAI_MODEL` 配（注意代理的 `limited` 是权限标签非模型名，需填真实 id 如 qwen-flash/deepseek-v4-flash/gpt-5.5）。

### 阶段 13 — v3：并行预测编码流 + 显著度门控全局工作空间（默认关、零回归）

诊断图 `diagrams/2026-06-23T010000` 画了"6 脑区并行 → Integrator 平均"，但代码是线性串行链、`affect_core` 顺序平均。本步据 30 篇神经科学文献（`notes/2026-06-25-…`）把图修正为**生物学忠实版**：并行的是**功能流（时间尺度×精度）**非解剖脑区，整合是**显著度门控 ignition 广播**非平均，快路是**亚符号生存信号**非"恐惧"。设计见 `PRP/affective-expression-v3-parallel-workspace/design.md`。

- **理论裁决**：并行真实（杏仁核 many-roads[Pessoa&Adolphs]、丘脑枕-杏仁核人类实证、三网络、并行预测编码、DA 并行广播），但"1:1 脑区=模块"被否（Pessoa 双重竞争 / Barrett 简并无指纹 / CPM 评价检查其实序列），整合须 GNW ignition（Dehaene）+ 显著网络门控（Menon）。
- **`affect_math.py`（纯新增）**：`fast_survival_prior`（上丘-枕-杏仁核捷径/生存回路，粗效价+高唤醒、低固定精度）+ `stream_salience`（SN 门控分数 |μ|·Π̄）+ `ignite`（salience≥阈点燃、不空播）+ `precision_reconcile`（Kalman 式精度加权再入，替代固定 0.5）+ `SURVIVAL_PRECISION`/`SALIENCE_THRESHOLD`/`AROUSAL_GAIN`/`LANG_BASE_PRECISION`。**不动** `gaussian_fuse`/`fuse_terms`/`reconcile_affect`。
- **`affect_core.py`**：`workspace_enabled` 分支——并行流 [survival, appraisal, value, (mood)] → `ignite` → `fuse_terms` → 采样，写 `ignited_streams`/`affect_precision`；唤醒增益（NE 精度调制）抬高评价/价值流投票权。默认关时返回 dict 与 trace 逐字同 v1/v2。
- **`language.py`**：`workspace_enabled` 时回路重写用 `precision_reconcile`（高精度内核抗语言拉拢）；否则 `reconcile_affect` 不变。
- **贯通**：`state.py` 加 `workspace_enabled`/`ignited_streams`/`affect_precision`；`runner.run` 形参贯通 + 轨迹输出点燃流；`main.py --workspace` 验证入口。
- **验证**：`pytest` **158 passed, 5 skipped**（+9 新）；`ruff`/`mypy`(41 源) 干净；`python main.py --workspace` 实跑显示流差异化点燃（巨响→survival+appraisal、offer→三流齐燃、value 对无回报刺激被门控）；workspace+language 精度再入端到端不崩、高精度内核抵抗漂移。

### 阶段 14 — 存储层后端拆分 + 交互对话入口（情感引擎 ⊗ LLM 耦合，真机验证）

承接遍历观察（`graph_store.py` 556 行偏重）+ 用户现阶段目标「验证情感引擎与 LLM 输出耦合、需要能真正交流对话的入口」。两件事：

- **`graph_store.py` 拆分（零破坏重构）**：556 行拆成 `src/storage/backends/deterministic.py`（InMemory/Sqlite/Neo4j + StoredFact + GraphStore 协议）+ `backends/semantic.py`（SemanticStore 协议 + Graphiti/SqliteVector + `_cosine`/`_build_graphiti`/`_coerce_dt`/`_group_id`）；`graph_store.py` 收为**门面**（再导出 + `__all__`）。**工厂 `build_graph_store`/`build_semantic_store` 及其探测 `_neo4j_store`/`_graphiti_store`/`_sqlite_vector_store` 刻意留在门面同模块**——测试 `monkeypatch.setattr(gs, "_neo4j_store", …)` 后调工厂须同命名空间方能命中（拆到子模块会令 monkeypatch 失效）。上层导入路径不变，零回归。
- **交互对话耦合入口**（一直缺的环节）：情感引擎已成、LLM 已接，但只有批处理 demo、无对话循环。补两块——①**评价桥**：`OpenAILanguageModel.appraise_text(text)`（复用与生成解耦的独立 VAD 反推，把用户每句话读成 (v,a)）；②**多轮会话基元** `runner.ConversationSession`（建图/checkpointer 一次、逐轮 `step`，运行态 mood/value_table **跨轮持久** → 情绪连续/A.7 滞后在对话里显现）。`main.py --chat`：你说一句 → 评价桥读情绪 → 喂引擎演化 e*/mood → 语言层生成带情绪回应；有 LLM key 走真模型，缺 key 回退词典评价+模板语言。`main.py` docstring 从「临时脚本」**转正**为官方入口。
- **真 LLM 端到端已验证** ✅（qwen-flash，`main.py --chat`）：升职→狂喜·seeking、被羞辱→暴怒·rage 情绪对路；连续负面后 mood valence 累积下行（+0.12→-0.29），末轮仅轻微负面输入(-0.30) 仍因负盆 mood 维持暴躁回应——**滞后/历史依赖在真实对话里显现**，坐实情感引擎 ⊗ LLM 耦合。
- **落点**：`runner.py`（抽 `_state_to_entry` 共用 + `ConversationSession`）、`language_openai.py`（`appraise_text`）、`main.py`（`--chat` + 日志静音 + docstring 转正）、`storage/backends/`（新子包）、`graph_store.py`（门面）。新测 `test_conversation.py`（会话 mood/value 跨轮持久 + appraise_text fake-client）。
- **验证**：`pytest` **162 passed, 5 skipped**（+4 对话测试；拆分零回归）；`ruff`/`mypy`(45 源) 干净。

### 阶段 15 — 对话情绪重做（响应·积累·多样·部分随机·有边界，按真机反馈两轮迭代）

阶段 14 的 chat 真机暴露问题，两轮迭代修到位：

- **第一轮（连贯+落库+不极端）**：补 `converse(history, …)`（带完整对话历史 → 连贯、能答"我刚说了什么"）+ `ConversationLog`（stdlib sqlite，transcript 落盘、重启重载 → 跨重启记忆）+ `_sqlite_saver().setup()` 幂等建表（修 sqlite checkpointer 首用 no-such-table）。表达先试**慢变 mood 主导 + 阻尼**。
- **第二轮（情绪要真积累且响应）**：真机发现 mood 主导**反而锁死**——A.7 双稳从正盆自锁、单调爬到 +1，被持续辱骂也不变负（情绪不响应、一成不变、回应只能维持"好好先生"）。根因：双稳 `gain·tanh(k·m)` 自放大 + 过度阻尼。改为：
  - **泄漏积分情绪**：`mood_step(self_gain=0, inertia=0.7, drive=0.3)` + 小噪声（chat 关 `mood_enabled`，不用双稳）→ 持续负面**逐步累积变负、会动怒**，响应输入、不锁死、部分随机。减阻尼（`goal=v`、intensity 不再压到 0.6）。
  - **多样情绪词**：映射 `affect_label`（8 扇区×强度，恼火/愤怒/低落/愉悦…）非单一 v/a。
  - **有脾气不讨好**：`_CONVERSE_SYS` 强化「负面时绝不一味道歉/自责/讨好/当出气筒，可不耐烦/设边界/冷淡/回怼，有自尊」+ converse 温度抖动。
  - **`ZERO_CHAT_THREAD`** 切独立会话/重置；累积情绪也落 `ConversationLog` meta、重启续上。
- **关键教训**：LLM 会**模仿历史里自己的回合**——旧历史若全是讨好回应，新回合继续讨好；干净历史下强化提示词才生效。
- **真机验证**（qwen-flash，干净历史）：辱骂升级时情绪逐步变负（平静→专注→恼火→愤怒）、回应有真脾气与边界（"不想惯着你这种说话方式"/"先学会怎么跟人说话"），连贯、跨重启续上。
- **落点**：`language_openai.py`(`converse`/`affect_label` 多样词/温度抖动/`_CONVERSE_SYS` 抗讨好)、`main.py`(`_run_chat` 泄漏积分 + `ConversationLog`(transcript+情绪) + `ZERO_CHAT_THREAD` + 日志静音)、`checkpointer.py`(`setup()`)。新测 `test_conversation.py`(+3：converse 带历史/泄漏积分响应不锁死/transcript+情绪落库)。
- **验证**：`pytest` **165 passed, 5 skipped**（+3）；`ruff`/`mypy`(45 源) 干净。

## 成果与验证

- **测试**：`pytest` 165 passed, 5 skipped（含 ml 缺 torch 跳过 2 + Graphiti 实机 smoke neo4j/kuzu 各跳过 1 + steering 实机 smoke 缺 ZERO_STEER_MODEL 跳过 1；本机有 sentence-transformers 故句向量 6 测全跑）；`ruff check`/`ruff format`/`mypy`(45 源文件) 干净。
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
- 分支 `chore/temp-main-launcher`（阶段 10）：文本输入侧句向量升级——`STTextAffectRegressor`（冻结 MiniLM + MLP 头）+ `emobank_st`/`train_text_affect_st` + `test_emobank_st`，词袋版零回归并存作基线；实测 loss 降 64% + 跨域口语/商业体裁判对；新 `nlp` extra（`sentence-transformers`）。
- 分支 `docs/ravdess-prosody-done`（阶段 11，PR #16 已合 main）：RAVDESS 韵律 + WESAD 生理真实数据实跑记录（PROGRESS 待办标 3/3）。
- 分支 `docs/readme-weights-v0.2`（阶段 11，PR #17 已合 main）：README 同步三通道实跑 + 句向量升级 + `weights-v0.2`（real-data trained）release 发布；ai-docs 本地知识层同步。

## 待办（需外部介入或独立轨道）

- **放数据跑真实训练（三通道全部实跑 3/3）**：**EmoBank 文本**（词袋 loss 0.016 / 句向量升级 0.0056、跨域更稳，见阶段 10）、**RAVDESS 韵律**（Zenodo 免登录、全量 1440 条，loss 0.126→0.026，pitch 随 arousal 单调上升的真实声学映射）、**WESAD 生理**（uni-siegen sciebo 全量 15 受试者 1474 窗，loss 0.053→0.024，stress 心率 92.7bpm/皮电最高 vs meditation 67.5 的应激→自主神经激活映射）均已实跑；脚手架（loader + `train_*`）在三类真实数据上验证可用。各通道 speech_rate/energy、体温等弱区分项属 loader 特征代理的尺度问题，非模型问题。
- **接真实后端**：本地已上 SQLite 落盘 + env 后端工厂；**Neo4j GraphStore 适配器已实现**（`Neo4jGraphStore` 裸 Cypher 保时序失效语义、`build_graph_store` 加 `neo4j` 分支 + 缺驱动告警回退，compose 已切 neo4j 后端）；**Postgres saver 已加固**（持显式长连接 + `autocommit/prepare_threshold/dict_row`，避开新版 `from_conn_string` 是 context manager、退出即关连接的坑）。**待真机验证**：在有 Docker 的服务器 `docker compose up` + 装 `db` extra，跑通 Postgres 跨重启恢复运行态 + Neo4j 时序语义（本机无 Docker，集成用例 `importorskip` + 连接探测优雅跳过）。**Graphiti 已深度集成（阶段 8）**：作为与确定性 GraphStore 并存的语义记忆侧信道接入（`SemanticStore`/`GraphitiGraphStore`/`write_episode`/`recall`，富 episode → 语义召回 → 语言层检索），`graphiti` extra、`ZERO_SEMANTIC_BACKEND=graphiti` 门控、默认关。**本地验证路径已就绪（阶段 9）**：`ZERO_GRAPHITI_DB=kuzu`（嵌入式、无 Docker/无服务）+ OpenAI 兼容 key，跑 `python -m scripts.verify_graphiti_local` 即可在本机验证闭环；**待用户带 LLM key 实跑确认**（本机无 key）。
- **扩 Worker 角色**：已加 MemoryRecall / Mood；可继续按 `/new-agent` 增加。

## 下一步路线图（阶段 15+ · AI 仿生人）

> 详见 [notes/2026-06-25-roadmap-bionic-human.md](notes/2026-06-25-roadmap-bionic-human.md)（定位确认 + 现状缺口 + 本地落库设计 + 学术依据）。**已定**：chat 默认翻为**本地落盘**（SQLite 三件套：运行态 + 确定性图谱 + 语义经历），零依赖内存档保留为 opt-in。

**记忆主线（近期，逐级依赖）**
- **阶段 15 — 本地持久化转正 + 对话经历入库 + chat 记忆闭环**：具身默认档（运行态 SQLite + `SqliteGraphStore` + `SqliteVectorStore`）；Supervisor 任务完成节点把「用户原话 + 仿生人回应 + e\*/mood/点燃流 + 时间」写成情景记忆（`SESSION` + 自传 `USER`）；chat 默认 `recall_enabled` → 仿生人「记得过去交流」。
- **阶段 16 — 自我模型 + 关系记忆**：稳定人格（大五 / PAD）偏置先验 + mood 盆深；对每个交流对象的关系 / 熟悉度记忆（回灌已有 `recalled_disposition` 通路）。
- **阶段 17 — 记忆巩固与遗忘**：情绪加权巩固（McGaugh）+ 遗忘曲线（Ebbinghaus），离线「睡眠」批处理 —— 绝不每条消息触发（守记忆节流红线）。

**多模态主线（中远期，独立轨道）**
- **阶段 18 — 多通道输入感知**：FER / SER / HRV / 文本编码器并行 → `fuse_terms` 精度融合（对称补全输入侧；「多网络并行」在此成形）。
- **阶段 19 — Live2D 适配器**（FACS AU → Cubism 参数）· **阶段 20 — 情感 TTS**（文本 + 韵律 → 表达性 TTS）。
- **阶段 21（可选）— 具身闭环 + 编排并行化**。

> 全程纪律：纯加法 / 鸭子类型注入 / torch·LLM·SDK 隔离 / env 门控 / 默认关 / 零回归 / 落地前固化 WebSearch 核验文献。
