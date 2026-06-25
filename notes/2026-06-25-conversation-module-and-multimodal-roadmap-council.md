# 科学家议会决策：对话模块扎实化 + 多模态扩展路线（第三次议会）

- 日期：2026-06-25
- 议题：先把 LLM 人机对话模块搞扎实、作为后续多模态（语音 SER / 视觉 FER / 生理 HRV）扩展的基座——可行吗？走 A（图原生）/ B（薄对话编排层）/ hybrid？
- 评审席位：CS（架构/可行性/红线，必到）· 数学（贝叶斯融合）· 神经（预测编码/层级）· 心理（评价理论/行为）；主持席综合
- 治理：全程只读、强制现场引文、不改代码、不介入运行时数据产生。本文件是议会唯一写动作。
- 背景：对话能跑，但对话的 LLM I/O（`converse`/`appraise_text`）、真正的记忆（`ConversationLog`）、两时间尺度情绪都在 LangGraph 图外（main.py `_run_chat`）；情感引擎在图内。已识别孤岛 `language_steering.py`、`text_affect` 输入通道。

## 三个架构候选
- **A 图原生**：对话回复/历史/情绪都进 LangGraph 节点。
- **B 薄对话编排层**：保留薄层走系统协议/分层。
- **hybrid**：感知侧图内（perception→affect_core，多模态 AffectEncoder 接 PerceptionAgent）+ 对话 LLM I/O 图外 REPL。

## 各席判定汇总

| 席位 | 判定 | 关键引用 |
| --- | --- | --- |
| CS（架构/可行性/红线） | **hybrid 忠实**；A 失真（interrupt 做逐轮聊天过重+重执行、history 进 state 撞红线）；B 有债。前置债：converse/appraise_text 不在协议 + 双存储并行 | LangGraph human-in-the-loop 官方文档 |
| 数学（贝叶斯融合） | 2 流（受约束 c·信源独立+精度加权）忠实；**N≥3 模态朴素精度相加失真**（double counting 超线性放大，ρ≈0.5 时 N=4 高估≈2.5x vs N=2 的 1.5x） | Ernst&Banks 2002 · Mihaylova 2017 · Julier&Uhlmann CI · Shams&Beierholm |
| 神经（预测编码/层级） | **先对话后多模态层级正确（忠实）**；平层精度加权简化（可防守，0.3 低值补偿层级）；语言是 top-down prior 调制器（非平级 evidence） | Barrett 2017 · Lindquist 2017 · Robins&Bhatt 2021 · Talsma 2015 |
| 心理（评价理论/行为） | 方向正确、可行；**模态冲突（反讽/掩饰）平层融合有界失真**（需高阶仲裁）；attitude 应进 AppraisalAgent 先验（非事后叠加） | Trautmann 2017 · Jessen&Kotz 2013 · Russell 1980 · Gross 2002 |

## 跨学科共识与张力

### 强共识（四席）：路线可行、方向正确、架构走 hybrid
"先对话后多模态"四席一致支持。神经席给出**主动论证**（非仅"先做简单的"）：语言/概念是情感生成模型的**最高阶 top-down prior**（Barrett 2017；Lindquist 2017 语言激活 aMPFC/vlPFC），从顶层 prior 模块建起在神经层级上是正确顺序。架构应走 **hybrid**：感知侧（含多模态 AffectEncoder）进 LangGraph 图内、对话 LLM I/O 留图外 REPL——**非纯 A 非纯 B**。A 失真（LangGraph interrupt 做逐轮聊天过重+节点重执行；对话 history 进 AffectState 撞"大对象进 state"红线）。

### CS「BLOCK」的精确化（主持席复核降级）
CS 描述"supervisor 每条消息触发三条记忆写入"。复核 `supervisor.py:22-49`：三条写入在图末端 `task_complete` 节点，且 `ConversationSession.step()` 粒度是"用户每句话 = 一次任务单元"，**技术上符合 memory-rules #1「任务完成节点」字面定义**，故**不是违红线的 BLOCK**。真实前置债是：① `converse`/`appraise_text` 是 `OpenAILanguageModel` 方法但**不在 `LanguageModel` 协议**（language.py:39 只有 `generate`），chat 直调真实类、破坏鸭子类型注入边界；② `ConversationLog`（transcript SQLite）与 supervisor `MemoryClient`（情感事件/episode）**两套并行存储**，同轮数据走两条路径，未来难统一查询/去重。

### 数学 ⟂ 神经 ⟂ 心理：在"多模态融合"上收敛到同一前置债
三席不同起点、同一结论——**当前 `fuse_terms` 平层独立流精度相加，在 2 流（受约束 c）可防守，扩 N>2 模态前必须升级融合算子**，须同时满足：
- （数学）模态间相关性处理避免 double counting（Covariance Intersection / 相关修正 / Bayesian Causal Inference）——情绪多模态高度相关（同一情绪事件的不同观测），非条件独立，N 增大后过度自信超线性放大。
- （神经）语言流应**动态调制**感官流精度（top-down prior），而非与感官流平级固定相加；感官流 ignition 应先于概念层。
- （心理）模态冲突（语言正面 + 语音/表情负面）需**高阶仲裁路径**（vmPFC 整合字面+社会情境），非加权平均；各模态应有差异化/动态基准精度（视觉>语音>文本）。

### 对话层语义债
- 协议边界（CS，工程）：converse/appraise_text 进协议。
- 情绪模型统一（心理）：chat 层 `emotion`/`attitude`（main.py REPL 变量，图外）与图内 `mood` 构念断层——attitude 目前只进 LLM prompt 着色、未进 `AppraisalAgent` 先验（`Stimulus.attitude_appeal`）。

## 决策

### 可接受的刻意简化（及代价）
| 简化 | 代价 | 依据 |
| --- | --- | --- |
| supervisor 每 stimulus（=每句话）flush 三条记忆 | 对话密集时写频略高 | 每 stimulus 完成=任务单元，符合 memory-rules #1 字面定义 |
| fuse_terms 精度加权在 2 流（OCC+text）信源独立下运行 | N>2 不可直接扩 | Ernst&Banks 2002 条件独立 MLE；议会二受约束 c |
| 语言流固定精度 TEXT_AFFECT_PRECISION=0.3 | 高质量文本低估其权重 | Friston 2005 允许固定精度作初始近似 |
| emotion/attitude 为图外 REPL 变量、不进 AppraisalAgent 先验 | attitude 只作 LLM prompt 着色、未真正参与评价融合 | 实现路径最短；可在阶段15/16 用 attitude_appeal 接入 |

### 必须改（前置债 / 失真）
1. **协议边界（前置·纯工程）**：`converse`/`appraise_text` 加入 `LanguageModel` 协议或建独立 `ConversationModel` 协议，使 chat 路径可注入/可替换/可单测。
2. **双存储分工（前置）**：明确 `ConversationLog`（transcript 运行态）vs supervisor `MemoryClient`（情感事件/长期 episode）各自职责，避免阶段15「对话经历入库」时进一步分叉。
3. **多模态融合算子（扩 N>2 前必须回议会专评）**：当前平层精度加权在多模态存在 double counting（数学）+ 层级混淆（神经）+ 冲突盲区（心理）三重失真。阶段18 启动前提交独立议会，设计 Covariance Intersection / 动态精度 / 冲突检测-仲裁。
4. **attitude 接入 AppraisalAgent 先验（心理）**：attitude 作评价先验（进 `Stimulus.attitude_appeal`），非仅 e* 后叠加。可在阶段15/16 记忆接入时一并处理。

### 悬而未决
1. `ConversationModel` 协议边界具体签名（扩展 `LanguageModel` 还是独立定义）——待 /engineer 工程草案确认。
2. supervisor 记忆写入粒度是否随会话模式调整（批处理 vs 对话不同节流）——有价值非阻塞。
3. attitude tuple → `Stimulus.attitude_appeal` 标量的归一化映射。

### 分阶段路线
```
前置必做（不依赖架构决策，现在即可动手 · 纯工程 /engineer）
  └─ 协议边界：converse/appraise_text 进协议或独立 ConversationModel 协议
  └─ 明确双存储分工：ConversationLog vs supervisor MemoryClient 职责

阶段15（/engineer 可直接落地）
  └─ chat 默认落盘 + 对话经历 episode 入库 + recall_enabled 默认开；同步厘清双存储分工

阶段16（/engineer，需对照心理席 attitude 约束）
  └─ attitude 进 AppraisalAgent 先验（Stimulus.attitude_appeal）+ 关系记忆/自我模型持久化

阶段18 前置（必须回议会专评，不可直接 /engineer）
  └─ 多模态融合算子设计议会：Covariance Intersection / 动态精度 / 冲突检测-仲裁
     （N>2 double counting 修正 + 语言 top-down 调制 + 模态冲突仲裁）

阶段18（议会专评后 /engineer 落地）
  └─ AffectEncoder 协议 + 各模态并行编码器 + 升级后融合算子
```

## 结论：NEEDS-CHANGES（前置修协议边界 + 厘清双存储）→ 修后 PASS
**现在可以启动"把对话模块搞扎实"**，主体路线（hybrid + 先对话后多模态）四席共识、神经学层级忠实、工程可行。两项前置债（协议边界、双存储分工）须同步处理。多模态融合算子在 N=2（受约束 c）下可防守继续推进对话层；**N>2 扩展前须专项议会评审**——这是清晰门槛点而非模糊约束。

## 引文（各席现场核验，附链接）
- Ernst, M. O. & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature* 415:429-433. [DOI:10.1038/415429a](https://doi.org/10.1038/415429a) — MLE 最优整合须条件独立。
- Mihaylova, L. et al. (2017). Distributed Multisensor Data Fusion under Unknown Correlation and Data Inconsistency. *Sensors* 17(11):2472. [PMC5713506](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5713506/) — 共享信源未建模→协方差低估（double counting）。
- Julier, S. J. & Uhlmann, J. K. (1997). A non-divergent estimation algorithm in the presence of unknown correlations. *Proc. ACC*. [DOI:10.1109/ACC.1997.609105](https://doi.org/10.1109/ACC.1997.609105) — Covariance Intersection。
- Shams, L. & Beierholm, U. R. (2010). Causal inference in perception. *Trends Cogn. Sci.* 14(10):425-432. [DOI:10.1016/j.tics.2010.07.001](https://doi.org/10.1016/j.tics.2010.07.001) — 贝叶斯因果推断整合。
- Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc. B* 360:815-836. [DOI:10.1098/rstb.2005.1622](https://doi.org/10.1098/rstb.2005.1622) — 精度加权预测编码；固定精度作初始近似。
- Barrett, L. F. (2017). The theory of constructed emotion. *SCAN* 12(1):1-23. [PMC5390700](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390700/) — 概念/语言是情感生成最高阶 top-down prior。
- Lindquist, K. A. et al. (2017). The role of language in emotion: a neuroimaging meta-analysis. *SCAN* 12(2):169-183. [PMC5390741](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390741/) · [DOI:10.1093/scan/nsw121](https://doi.org/10.1093/scan/nsw121) — 语言激活 aMPFC/vlPFC，top-down 调制器。
- Robins, D. L. & Bhatt, R. S. (2021). Crossmodal processing in the posterior superior temporal sulcus. *Front. Hum. Neurosci.* 15:627450. [PMC8022349](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8022349/) — pSTS 非线性门控。
- Talsma, D. (2015). Predictive coding and multisensory integration: an attentional account. *Front. Integr. Neurosci.* 9:19. [PMC4374459](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4374459/) — 注意≠整合。
- Trautmann, S. A. et al. (2017). Emotion perception from face, voice, and touch. *Trends Cogn. Sci.* 21(3):216-228. [PMC5334135](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5334135/) — 多模态情感整合行为证据 + 不一致冲突。
- Jessen, S. & Kotz, S. A. (2013). The temporal dynamics of processing audiovisual emotional incongruence. *Neuropsychologia* 51(9):1820-1825. [DOI:10.1016/j.neuropsychologia.2013.05.004](https://doi.org/10.1016/j.neuropsychologia.2013.05.004) — 模态冲突时序动力学。
- Russell, J. A. (1980). A circumplex model of affect. *JPSP* 39(6):1161-1178. [DOI:10.1037/h0077714](https://doi.org/10.1037/h0077714) — 环状 VA 空间。
- Gross, J. J. (2002). Emotion regulation: affective, cognitive, and social consequences. *Psychophysiology* 39(3):281-291. [DOI:10.1017/S0048577201393198](https://doi.org/10.1017/S0048577201393198) — 两时间尺度（emotion/attitude）心理学依据。

## 相关代码位置（只读）
- `src/orchestration/supervisor.py:22-49`（三条记忆写入，任务完成节点）
- `src/agents/language.py:39-50`（LanguageModel 协议，缺 converse/appraise_text）
- `src/agents/language_openai.py`（converse/appraise_text 实现，在协议外）
- `main.py:109-161`（ConversationLog，第二套存储）
- `main.py:205-249`（chat REPL 主循环，图外 LLM I/O 的现有 hybrid 实现）
- `src/orchestration/runner.py:126-199`（ConversationSession，感知侧图内）
