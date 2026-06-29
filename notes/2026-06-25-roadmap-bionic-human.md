# AI 数字人路线图：从「情感表达子系统」到「有长期记忆与经历的拟人体」

> **日期**：2026-06-25
> **性质**：工程规划（非实现）。在阶段 0–14 已有产物之上，确定项目定位、盘点真实缺口、设计「经历/记忆落本地库」、给出阶段 15+ 路线图。
> **约束**：严格遵守三层架构红线 + 模块解耦纪律（鸭子类型注入 / torch·LLM 隔离 / 默认关 / 零回归）；所有新机制须有学术材料支撑，落地前按本仓既有惯例（见 `notes/`）逐阶段固化 WebSearch 核验过的文献清单。

---

## 一、项目定位（确认）

**Zero = 一个 AI 数字人**：以**情感引擎**为内核、**LLM** 为语言外化，使人机交流带有近似人类的情感动力学，从而更拟人。

- **情感引擎**：数学（贝叶斯主动推断 / OCC 评价 / TD-RPE / 双稳心境）× 心理学（评价理论 / Gross 调节 / 大五人格）× 生物学（杏仁核多通路 / 自主神经 / 面神经双通路）× 神经科学（全局工作空间 ignition / 显著网络门控 / 预测编码）× 计算科学（多 Agent 编排 + 多网络并行）的融合，在计算层面近似人类情感反应。
- **「多 Agent + 多网络并行」的落点**：已不是设想——v3 工作空间（阶段 13）已实现并行情感流（survival/appraisal/value/mood）+ 显著度门控 ignition。本路线图把第二条并行轴补齐：**多通道感知的并行编码网络**（图像/心电/语音/文本各一张网，按精度融合），与已有并行情感流共同构成「多网络并行 → 精度加权整合」的生物学忠实结构。
- **当前能力（已验证）**：纯文字交流已带真实情感动力学——`main.py --chat` 真 LLM（qwen-flash）端到端验证：升职→狂喜·seeking、受辱→暴怒·rage 情绪对路；连续负面后 mood 累积下行、轻微正向也拉不出负盆（A.7 滞后在真实对话里显现）。
- **目标演进**：① 近期——把**记忆/经历真正落本地库、长期留存**，让数字人「记得过去的交流」；② 中期——**多通道输入**（视觉图像 / 心电 / 文本 / 表情 / 语气）；③ 远期——**多模态输出**（文本 + Live2D 形象表现 + 情感 TTS 内容与语气）。

---

## 二、现状盘点（已建成 vs 真实缺口）

### 已建成（阶段 0–14，测试 162 passed / 5 skipped）

| 层 | 已有 |
| --- | --- |
| 编排 | `build_graph` 10 节点 + 条件边双向回路；`run` / `ConversationSession`（多轮 mood/value 跨轮持久） |
| Agent | perception/appraisal/value/affect_core/mood/regulation/expression + language/language_openai + memory_recall；`affect_math` 数学内核（OCC/TD/精度/高斯融合/mood 双稳/工作空间 survival·salience·ignite·精度再入） |
| 真网络化 | 输出侧解码器 ExpressionDecoder/Prosody/Physiology/Facs + 输入侧 TextAffectRegressor/STTextAffectRegressor；三通道（文本/韵律/生理）已在真实公开数据上实跑（weights-v0.2） |
| 记忆 | `MemoryClient`：确定性 `write/query`（时序失效）+ 语义 `write_episode/recall`；显式 scope + 任务完成节流 |
| 存储 | 运行态 Checkpointer（InMemory/SQLite/Postgres）；确定性图谱（InMemory/SQLite/Neo4j）；语义侧信道（SqliteVectorStore/Graphiti），env 选后端 |

### 真实缺口（对「数字人 + 长期记忆/经历」而言）

1. **默认全内存 → 重启即失忆**。`build_graph_store / build_checkpointer / build_semantic_store` 默认 InMemory/None，`main.py --chat` 走默认 → **对话经历不跨进程留存**。基础设施齐备，缺一个「具身持久化默认档」的接线决策。
2. **存的是情感事件摘要，不是「经历」本身**。Supervisor 只写 `event=X affect=(v,a)` 和 disposition；**用户原话 + 数字人回应 + 时间 + 情感上下文**这种自传式情景记忆没有入库。「经历」= 可回忆的对话情景档案，当前缺失。
3. **没有「自我」与「关系」的持久化**。`user_id="default-user"`；没有数字人自身的稳定人格/传记，也没有「对每个交流对象的关系记忆」（熟悉度/历史情感）。
4. **没有记忆巩固与遗忘**。事实只在 (scope,key) 撞键时失效、episode 无限堆积；缺生物学上的**情绪加权巩固**（情绪事件记得更牢）与**遗忘/衰减**。
5. **chat 入口未闭合记忆读环**。`ConversationSession` 默认 `recall_enabled=False`，交互入口里 MemoryRecall 不读、语义召回不进语言层——「记得」在 demo 里没真正打开。
6. **感知是占位 + 单模态**。`PerceptionAgent` 直接取 OCC 维度作占位特征；真网络化的**编码器只有文本输入侧**，图像/心电/语音输入侧、Live2D/TTS 输出侧均未接。

> 关键观察（对称性）：已有 Prosody/Physiology/Facs 都是**输出解码器** `(v,a)→通道特征`；多通道**输入**需要对称的**编码器** `通道特征→(v,a)`。二者结构对称、可复用同一套注入协议与 torch 隔离纪律——这是多通道扩展低成本、低耦合的根因。

---

## 三、本地数据库 / 长期记忆 / 经历——落库设计（核心）

**原则**：复用既有 SQLite 后端，**无图库、无服务、无 Docker**（用户要求「无额外多余负担」、本地优先）。运行态与长期记忆**分离存储**（红线）；secrets 只走 `.env`。

### 落库映射（规划）

| 数据 | 记忆学类别 | 后端（已存在） | scope / key 约定 | 写入时机 |
| --- | --- | --- | --- | --- |
| mood、value_table、回路态 | 运行态（工作记忆） | SQLite Checkpointer (`ZERO_CHECKPOINT_BACKEND=sqlite`) | `thread_id` | 每轮（Checkpointer 自动） |
| 对话经历（用户原话 + 回应 + e\* + mood + 点燃流 + 时间） | **情景记忆**（episodic，Tulving 1972） | SqliteVectorStore（语义召回）+ SqliteGraphStore（结构化检索） | `SESSION`=本次会话 / `USER`=自传式长期 | 任务完成节点（节流） |
| 长期情绪倾向 / disposition | 语义记忆（semantic） | SqliteGraphStore（时序失效） | `USER`=交流对象 | 任务完成节点 |
| 自我人格 / temperament / 传记 | 自我模型 | SqliteGraphStore | 新约定 `key="self"`（或新 scope） | 初始化 + 巩固时更新 |
| 对每个人的关系 / 熟悉度 | 关系记忆 | SqliteGraphStore | `USER`/`GROUP` per-interlocutor | 任务完成节点 |
| LLM key / 后端连接串 | 配置（非记忆） | `.env`（gitignore） | — | 不入库（红线） |

要点：**全部用已有的 SQLite 三件套即可承载**，不引入新存储依赖。把「默认全内存」翻成一个**具身持久化默认档**（chat 模式默认落盘，零依赖内存档保留为 opt-in）就解决了缺口 1/5 的一大半——这是阶段 15 的主体。

---

## 四、分阶段路线图（阶段 15+，承接 PROGRESS 编号）

> 每阶段沿用本仓既有纪律：**纯加法 / 鸭子类型注入 / torch·LLM 隔离 / env 门控 / 默认关 / 零回归 / 落地前固化文献**。编号衔接 PROGRESS.md 的阶段 14。

### 阶段 15 — 本地持久化「转正」+ 对话经历入库 + chat 记忆闭环〔最高优先，直答「数据入本地库 / 长期记忆 / 经历」〕
- **落点（规划）**：①「具身默认档」——chat 默认走 SQLite（运行态）+ SqliteGraphStore（确定性事实）+ SqliteVectorStore（语义经历），路径由 `.env` 注入；零依赖内存档保留为显式 opt-in。②**对话经历 episode**——在 Supervisor 任务完成节点把「用户原话 + 数字人回应 + e\*/mood/点燃流 + 时间戳」写成情景记忆（`SESSION` + 自传 `USER`），不改节流与显式 scope 红线。③ chat 默认 `recall_enabled=True`，让 MemoryRecall 读 disposition + 语义召回 `recalled_context` 进 LanguageAgent——数字人「记得上次聊了什么」。
- **学术依据**：情景记忆 vs 语义记忆（Tulving 1972, 1983）；自传式记忆。
- **耦合纪律**：写仍只在任务完成节点（memory-rules #1）；存储层不上调；运行态不写图谱（红线 #3）。

### 阶段 16 — 自我模型 + 关系记忆（人格持久化）
- **落点（规划）**：① 稳定自我——temperament（PAD 基线心境）/ 大五特质 / 简短传记，持久化为 `key="self"`；人格参数**偏置** mood 盆深与 appraisal/arousal 增益（接已有 mood/精度机制，不新开热路径）。② 关系记忆——对每个交流对象累积情感史/熟悉度/信任，回灌 `recalled_disposition`（已有通路）。
- **学术依据**：大五 / HEXACO（Costa & McCrae；Ashton & Lee）；PAD temperament（Mehrabian 1996）；情感倾向。
- **耦合纪律**：人格只调先验/精度，不碰 `gaussian_fuse`/`fuse_terms` 内核；默认中性人格 → 零回归。

### 阶段 17 — 记忆巩固与遗忘（生物学忠实的记忆动力学）
- **落点（规划）**：① **情绪加权巩固**——高 |e\*|/高唤醒事件保留更强（杏仁核调制海马巩固）。② **遗忘/衰减**——低显著旧 episode 随时间衰减（遗忘曲线）。③ 巩固为**离线/节流的「睡眠」批处理**（会话结束或维护脚本触发），把 `SESSION` 经历择优固化为 `USER` 长期——**绝不每条消息触发**（红线 #1）。
- **学术依据**：情绪记忆调制（McGaugh 2000, 2004）；系统巩固（Squire & Alvarez 1995）；重放（Wilson & McNaughton 1994）；遗忘曲线（Ebbinghaus 1885）。
- **耦合纪律**：巩固是记忆层内的离线作业，不进编排热路径；衰减用时序失效语义表达，不物理删事实（红线 #4）。

### 阶段 18 — 多通道输入感知（对称补全输入侧编码器）〔中期目标：视觉/心电/文本/表情/语气输入〕
- **落点（规划）**：把 `PerceptionAgent` 泛化为**多模态精度加权融合**；定义 `AffectEncoder` 协议（输入侧，对称于 `ChannelDecoder`），按模态注入并行编码器，各吐 `(μ, 精度)`，用既有 `fuse_terms` 融合后喂 appraisal/工作空间：
  - 文本 → (v,a)：复用 `STTextAffectRegressor`（已有）。
  - 语气/语音 → (v,a)：语音情感识别（SER）编码器。
  - 表情/视觉图像 → (v,a)：面部表情维度识别（FER）编码器。
  - 心电 ECG → (v,a)：HRV→自主神经唤醒/效价编码器。
- **「多网络并行」在此真正成形**：每模态一张网并行编码 → 精度加权融合，与 v3 并行情感流同源。新 `vision`/`audio`/`physio-in` extra，torch 隔离、默认关、零回归。
- **学术依据**：维度化多模态情感（AffectNet circumplex, Mollahosseini 2017；IEMOCAP, Busso 2008；MSP-Podcast；CMU-MOSEI, Zadeh 2018；WESAD, Schmidt 2018；DEAP, Koelstra 2011；HRV-情感, Appelhans & Luecken 2006）；贝叶斯/精度加权多感觉整合（Ernst & Banks 2002；主动推断, Friston）。

### 阶段 19 — Live2D 表现输出适配器〔远期目标：操控 Live2D〕
- **落点（规划）**：`ExpressionRenderer` 适配器，消费 `ExpressionAgent` 已产出的 FACS AU + 双通路（真笑/假笑）→ 映射 Live2D Cubism 参数（口/眼/眉等），按强度驱动；流式更新。下游适配器，**不耦合进内核**，默认关。
- **学术依据**：FACS（Ekman & Friesen 1978）；AU→blendshape/ARKit 映射；Live2D Cubism 参数模型（厂商规范，工程实现层）。

### 阶段 20 — 情感 TTS（内容 + 语气）输出〔远期目标：TTS 内容与语气〕
- **落点（规划）**：`SpeechRenderer` 适配器，消费 LanguageAgent 文本（内容）+ ProsodyDecoder 韵律（pitch/energy/rate，语气）→ 表达性 TTS（SSML prosody / 风格条件化）。默认关、可注入不同 TTS 后端。
- **学术依据**：表达性/情感 TTS（Global Style Tokens, Wang 2018；Tacotron2 韵律；情感 TTS 综述）；VAD→韵律映射（承接阶段 11 RAVDESS「pitch 随 arousal 单调」实测）。

### 阶段 21（可选）— 端到端具身闭环 + 编排并行化
- **落点（规划）**：实时多模态闭环（多通道感知 → 演化 → 多模态表达）；视情把感知编码器做成 LangGraph 真并行 fan-out（Supervisor 分发并行节点）；实时短期态可上 Redis。此处「多 Agent + 多网络并行」全量落地。

### 依赖路径与优先级
```text
阶段15(持久化+经历入库) ──→ 阶段16(自我/关系) ──→ 阶段17(巩固/遗忘)        ← 记忆主线（近期，逐级依赖）
       │
       └─（独立并行轨道，不阻塞记忆主线）
阶段18(多通道输入) ──→ 阶段19(Live2D) / 阶段20(TTS) ──→ 阶段21(具身闭环)     ← 多模态主线（中远期）
```
近期聚焦记忆主线（15→16→17），多模态主线可作独立轨道按需推进。

---

## 五、贯穿原则（自查清单）

- **三层依赖单向**：编排 → 记忆 → 存储；新编码器/渲染器经协议注入，存储/记忆层不上调。
- **记忆写入节流 + 显式 scope**：经历/巩固只在任务完成或离线批处理写，不在每轮/每步写；读写带显式 `Scope`。
- **运行态 vs 长期记忆分离**：mood/value 进 Checkpointer，不入图谱；图谱事实带时序、失效非删除。
- **耦合纪律**：核心编排/数学内核不依赖 torch/LLM/TTS/Live2D SDK；所有重依赖进 optional extra，鸭子类型协议注入，默认关、零回归。
- **学术可追溯**：每阶段实现前固化一份 WebSearch 核验的文献 note（对齐 `notes/2026-06-2x-*`），本路线图的 author/year 为方向锚点、非最终引用。
- **无多余负担**：本地优先用 SQLite 三件套承载全部记忆，不为「长期记忆」强上图库/容器。

---

## 六、决策记录（2026-06-25 已定）

1. **持久化默认档** ✅ 定为「默认落盘」：chat 默认走 SQLite 三件套（运行态 + 确定性图谱 + 语义经历），零依赖内存档保留为 opt-in。阶段 15 据此设计。
2. **近期范围** ⏳ 先完成整体规划；记忆主线（15→17）为近期重心，多模态主线（18+）作独立轨道。用户另有并行窗口在推进，实际进度待后续合并评估。
3. **定位同步** ✅ README / PROGRESS 顶部定位已上调为「AI 数字人」全景，并把本路线图接入 PROGRESS 待办。
