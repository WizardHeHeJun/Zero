# Harness 工程计划与工程评估

> **日期**：2026-06-25
> **性质**：规划 + 评估。落实目标「按上文思路搭建 harness 工程 + 制定整个工程计划 + 评估工程」。
> **承接**：`notes/2026-06-25-roadmap-bionic-human.md`（阶段路线图）、`notes/2026-06-25-science-council-design.md`（科学家议会）、跨会话记忆 `analysis-results-first-no-intervene`（结果优先、不介入数据产生）。

---

## 一、Harness 工程逻辑（"造仿生人的机器"怎么运转）

harness 不是产品，是**用来建造 AI 仿生人的机器**：三层资产 + 一条主回路 + 一组治理原则。

### 1.1 三层资产

| 层 | 资产 | 职责 |
| --- | --- | --- |
| 知识层 | `CLAUDE.md` · `.claude/rules/`（glob 注入）· `ai-docs/`（模块三件套/catalog/pitfalls）· `notes/`（研究·设计固化） | 渐进式加载当前任务相关知识，喂红线与上下文 |
| 工作流层 | `/dev` 路由 · PRP 四阶段（refine-prd→generate-prp→validate-prp→execute-prp）· `/generate-doc`·`/learn`·`/gc` | 把需求按复杂度路由到落地，并回写知识层 |
| Agent/技能层 | 产品 Worker（`src/agents/`）· **科学家议会 6 席**（研究/评审）· **软件工程师团队 5 角色**（工程落地）· `code-reviewer`（独立审查门）· skills · hooks | 实现 + 审查 + 跨学科评审 + 工程落地 + 自动化 |

### 1.2 主回路（端到端，议会门已接入）

```text
需求 → /dev 判复杂度
  ├─ 简单 → 直接做 → 测试/lint
  ├─ 中等 → Plan 模式 → 实现 → 测试/lint
  └─ 复杂 → /refine-prd → /generate-prp
                              └─【科学家议会设计门】涉跨学科算法时 /science-council
                                  → 各席 忠实/简化/失真 + 强制引文 → moderator 综合 → 落 PRP/design.md
                          → /validate-prp（校验含「学术忠实性」项）
                          → /execute-prp【软件工程师团队落地：架构师定落点→实现→测试→集成】
                              →【code-review 独立审查门】项目级宏观审查（记忆时机/层越界/节点契约/async）
                          → /generate-doc 同步知识层 → /learn 沉淀 → /gc 健康度扫描
```

新接入点（本轮搭建）：`/generate-prp` 加「跨学科评审门」、`/validate-prp` 加「学术忠实性」校验项——把议会焊进复杂特性的设计闸口。

### 1.3 治理原则（贯穿全机器）

1. **结果优先、不介入数据产生**：议会等分析机制以项目已产出结果为起点、只读、零反馈，绝不下场生成情绪/记忆/语言（见 `science-council-design` §一）。
2. **学术可追溯**：每个跨学科决策强制现场 WebSearch 引文，固化进 notes/PRP——议会把这条从"自觉"变"机制"。
3. **架构红线**：三层单向依赖、确定性热路径（torch/LLM-free、可复现、零回归）、记忆节流+显式 scope、重依赖进 optional extra + 鸭子类型注入。
4. **harness 项目中立、此处实例化**：机器本身可复用，当前实例化为"造 AI 仿生人"。

---

## 二、整个工程计划（路线图阶段 × harness 流程）

把路线图阶段 15–21 每步映射到 harness 流程：走哪条路径、是否过议会门、动哪些模块、怎么验证。

### 2.1 阶段 → 流程映射

| 阶段 | 路径 | 议会门 | 主要模块 | code-review | 关键验证 |
| --- | --- | --- | --- | --- | --- |
| 15 持久化转正 + 经历入库 | PRP | 否（纯工程，但**记忆纪律**是 review 重点） | storage·memory·orchestration | 必 | 跨重启留存 + 写入节流 + 显式 scope |
| 16 自我模型 + 关系记忆 | PRP | **是**（心理：人格-情绪） | memory·agents(appraisal/mood) | 必 | 人格只调先验/精度、默认中性零回归 |
| 17 记忆巩固 + 遗忘 | PRP | **是**（生物 McGaugh × 数学衰减） | memory·storage | 必 | 离线"睡眠"批处理、不进热路径、时序失效非删 |
| 18 多通道输入感知 | PRP | **是**（全学科：FER/SER/HRV/精度融合） | agents(perception/models/datasets) | 必 | torch 隔离、默认关零回归、`fuse_terms` 融合 |
| 19 Live2D 渲染器 | PRP | 部分（生物 FACS→参数） | agents(expression renderer) | 必 | 下游适配器、不耦合内核 |
| 20 情感 TTS 渲染器 | PRP | 部分（韵律→prosody 映射） | agents(speech renderer) | 必 | 同上、承接 RAVDESS 实测 |
| 21 具身闭环 + 编排并行 | PRP | **是**（神经：并行/工作空间） | orchestration | 必 | 隔离回归、并发控制 |

### 2.2 排序与闸口

- **记忆主线**：15 → 16 → 17 串行（逐级依赖），近期重心。
- **多模态主线**：18 →（19 / 20）→ 21 独立轨道，不阻塞记忆主线。
- **闸口规则**：凡涉跨学科建模的阶段（16/17/18/21），先过 `/science-council` 设计门再定稿 PRP；纯工程阶段（15）跳过议会、但 code-review 加严。

### 2.3 首个试点

阶段 17（记忆巩固/遗忘）作议会第一炮：**以引擎已产出的记忆/情绪结果为分析起点**评审建模（McGaugh 情绪加权巩固 × Ebbinghaus 遗忘曲线 × 工程落库），既验议会又推路线图，天然符合"结果优先"。

---

## 三、工程评估（2026-06-25 结构性快照）

> **方法声明（诚实）**：本评估基于 PROGRESS/README/代码结构/git 状态的**结构性快照，未跑测试**——因 `src/` 正被并行窗口改动（11 个文件 mid-flight：chat 持久化 `data/*.sqlite3`、`converse` 历史、`checkpointer`/`language_openai` 改动）。此刻跑测试可能误判。待并行工作 commit/settle 后应跑 `pytest` 出权威快照（且可交 `/science-council` 评估，符合"以结果为起点"）。

### 3.1 现状盘点（git 实测）

- **产品 `src/`**：44 个 `.py`；阶段 0–14 完成；**阶段 15 持久化进行中**（并行窗口，11 文件 mid-flight）。
- **测试**：32 个测试文件（session-start 报 162 passed / 5 skipped；mid-flight 未重测）。
- **harness**：rules 6 · commands 11（+`science-council`）· agents 7（`code-reviewer` + 科学家 6 席）· skills 6 · ai-docs 四模块三件套 + catalog + pitfalls；notes 新增 roadmap / science-council / 本 plan。
- **跨会话记忆**：新增 `bionic-human-direction`、`analysis-results-first-no-intervene`。
- 注：`.claude/` 与 `ai-docs/` 为 gitignored 本地 harness 层，故本轮 harness 搭建不出现在 git status、与并行窗口零冲突。

### 3.2 成熟度评级

| 维度 | 评级 | 证据 | 缺口 |
| --- | --- | --- | --- |
| 架构分层 | 强 | 单向依赖测试 + rules 注入 | — |
| 数学内核 | 强 | 直接单测 + 边界钳制 | — |
| 真网络化 | 中-强 | 文本/韵律/生理 3/3 真实数据（weights-v0.2） | 表情 FACS 通道未喂真实数据；输入侧仅文本 |
| 语言耦合 | 强 | 真 LLM 验证（`--chat`/`--llm`） | — |
| 本地记忆/经历 | 中（上升中） | 持久化进行中（默认落盘） | 经历 episode / 自我 / 关系 / 巩固未做（阶段 16-17） |
| 真后端（图库/PG） | 中 | 适配器就绪（Neo4j/PG/Graphiti） | 待真机验证（本机无 Docker） |
| 多模态输入/输出 | 未启动 | 设计就绪（extension-guide 规划项） | 阶段 18-20 全部待建 |
| harness 完备度 | 强 | 知识层 + 工作流 + 评审议会闭环 | 议会未实战验证（待阶段 17 试点） |
| 学术可追溯 | 强 | notes 文献表 + 议会强制引文 | — |
| 可复现/测试 | 强（待重测） | rng_seed + 零回归 + 162 passed | mid-flight，需 settle 后重测 |

### 3.3 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 多窗口并发改 `src/` | 我只动 harness/docs（gitignored）不撞；评估用结构性快照，实测延后到 settle |
| 真机验证悬空（PG/Neo4j/Graphiti） | 本地优先 `SqliteVectorStore` 已闭环；真机待有 Docker 的服务器 |
| 议会成本/未实战 | 按需路由（非全员）+ 先拿阶段 17 试点 |
| 多模态 0 进度 | 独立轨道，不阻塞记忆主线；设计已就绪降低启动成本 |

### 3.4 就绪度结论

- **文字情感交流仿生人（当前目标）**：已达成并真 LLM 验证 ✅
- **长期记忆/经历落库（近期）**：进行中（阶段 15，并行窗口）
- **多模态（中远期）**：设计就绪、未动工
- **harness（造人机器）**：已成型——知识层 + 工作流 + 科学家议会评审门闭环，可驱动后续阶段
- **建议下一步**：① 并行窗口 settle/commit 后跑全量 `pytest` 出权威快照 ② 阶段 17 过议会门做首个试点 ③ 阶段 15 收口（对话经历 episode 入库）

---

## 四、本轮交付物清单（目标达成对照）

- ✅ **harness 工程搭建**：科学家议会 6 席 + `/science-council`（上一轮）+ 接入 PRP 设计门（`generate-prp` 评审门 / `validate-prp` 学术忠实性校验项，本轮）。
- ✅ **整个工程计划**：§一 harness 工程逻辑 + §二 路线图阶段 × harness 流程映射（含闸口/排序/试点）。
- ✅ **工程评估**：§三 结构性快照（现状/评级/风险/就绪度）+ 诚实的方法声明（实测待并行 settle）。
