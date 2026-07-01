# 科学家议会综合·第二轮：P3 关闭确认 / P4 arousal_gain 倒 U / Q5 关系多稳态语义裁决

> 承接首轮纪要 [seeking-attractor-council](2026-07-01-seeking-attractor-council.md)。P1（arousal_baseline 直流偏置根治）+ P2（habituation 习惯化）已落地并实测（阶段 31），`code-reviewer` PASS。本轮处理排后的三项。评审输入：[提案 round2](2026-07-01-p4-arousal-gain-q5-relationship-council-proposal-round2.md) + [Q5 工程方案](2026-07-01-q5-relationship-multistability-engineering-plan.md) + `src/agents/affect_core.py:45` + `src/agents/affect_math.py:36,182`。日期 2026-07-01（文件按 round2 命名区分首轮）。

---

## 一、触发与承接

**实测基准（工程师只读观测，议会以此为分析起点）**

- 修复态旋钮：`ZERO_INTENSITY_FLOOR=0` · `ZERO_AROUSAL_BASELINE=-0.08` · `ZERO_ATTITUDE_REVERSION_A=0.4` · `ZERO_HABITUATION_TAU=7`。
- 同一 24 轮日常友好对话：seeking 象限旧 6/24 → 修复 0/24；末轮 arousal 旧 +0.177 → 修复 +0.032。**seeking 吸引盆已拆除。**
- P3 观测（修复态 30 轮纯中性）：arousal 均值 -0.017，峰 +0.122 / 谷 -0.105，围绕 0 波动，无单调正漂移。
- P4 观测（修复态 24 轮强情绪）：arousal 峰 +0.278；进入 >0.5 危险区 0/24 轮。

**代码现状**（`affect_core.py:45`）：`arousal_gain = 1.0 + AROUSAL_GAIN * max(0.0, state.prior_mu[1])`，`AROUSAL_GAIN = 1.0`（`affect_math.py:36`），当前线性无界，负 arousal 段 gain 钳制到 1.0；倒 U 语义缺失。

---

## 二、各席判定汇总表

### P3：静息基线关闭确认

| 席位 | 判定 | 关键依据 |
| --- | --- | --- |
| 神经 | **PASS 关闭** | LC tonic 归一化后 arousal≈0 = 安静清醒基线可接受；gain=1.0 已隐式吸收清醒 tonic NE 基础放电，无需额外正 setpoint。 |
| 数学 | **PASS 关闭** | 修复态实测 a*≈-0.017≈0；`ATTITUDE_SETPOINT_A=0` + `arousal_baseline=-0.08` 的稳态解不含单调正偏置。 |
| **综合** | **PASS → P3 关闭** | 实测充分，神经席明确"gain=1.0 已隐式吸收清醒 tonic"，数学席验证 a*≈0。无需引入独立正静息值。 |

### P4：arousal_gain 倒 U

| 子问题 | 数学席 | 神经席 | CS | 综合裁决 |
| --- | --- | --- | --- | --- |
| **P4-a 形式** | 推 A（`1+c·a·(1-a)`）：单参数、斜率连续、a>0.5 自然反制正反馈、不引入新不动点 | 推 A 倒 U：α2A(高精度/PFC 激活) vs α1(低精度/PFC 抑制) 双侧衰减有分子机制，cap 只截高端不忠实 | PASS A（env 门控，不碰 `AROUSAL_GAIN` 常量） | **A 倒 U 为语义正确选择；cap 作短期防御先落；完整 A 排次优先** |
| **P4-b 范围** | c∈[1.5,3.0]（峰值 1.375-1.75，峰在 a=0.5） | c∈[1.5,2.5]（更保守忠实 LC 曲线） | env 控制即可 | **c∈[1.5,2.5]**（两席交集）· cap∈[0.3,0.6]（廉价 cap 阶段） |
| **P4-c 负段** | 可行但需下界钳制；暂不并行 | 现 gain=1 可接受，建议弱衰减 d∈[0.1,0.3]，与 Q7 deactivation 配套 | 未 BLOCK | **现行 gain=1.0 过渡可接受**；弱衰减 d∈[0.1,0.3] 作独立项与 Q7 配套，不并入本批次 |
| **P4-d 优先级** | 廉价 cap 防御足够、A 非紧迫（实测 a≤0.3，线性 vs 倒 U 差<15%，被多流稀释） | 实测 +0.278 远低于倒 U 右侧阈≈0.6；建议注释标触发判据（arousal>0.5 达 N=3/24 轮启动） | 廉价 cap PASS 可立即落地（`ZERO_AROUSAL_GAIN_CAP` 默认 None=关） | **当前批次 cap 先落（env 门控默认关）；完整 A 立项排后，待 arousal>0.5 实测触发** |

### Q5：关系多稳态语义

| 子问题 | 心理席（主裁） | 数学席 | CS | 综合裁决 |
| --- | --- | --- | --- | --- |
| **Q5-α 形式** | **C1 离散状态机**：Knapp 10 阶段无"陌生盆/亲密盆"双稳构念；C2 Gottman 双稳是"稳定幸福/不幸"非"陌生/亲密"，移植 pitchfork 引入假双稳失真 | C1 优于 C2：C2 参数可辨识性差（a·k·r 三参数冗余、势垒无法独立定标）；对称双稳无法表达升级≠降级不对称；Knapp 10 阶段 C2 需 5 层嵌套=15 参数 | C1 PASS（确定性标量门控）；C2 若 pitchfork 参数须 LLM 定→BLOCK | **C1 离散状态机。C2 永久排除**（心理构念无据+数学参数辨识差+不对称性缺失三重失真） |
| **Q5-β 判据** | 三类确定性触发：互惠自我披露/共同经历显著正事件/冲突未修复→降级；**不得含纯时间项** | C1 计数器门控数学良定义，"时间不触发升级"严格保证（对照 EWMA 每轮 rate·s 漂移）；N_up 陌生→初识 3-5/初识→朋友 10-15/朋友→亲近 20+/降级 N_down 2-5 | **PASS**（确定性标量门控）；警戒线"LLM 判跃迁→BLOCK" | **计数器门控 C1；事件：互惠自我披露/显著正事件/冲突降级；阈值见数学席；时间项禁止** |
| **Q5-γ 持久化** | 持久化是忠实关系构念的**必要条件非可选** | — | 成本真实（写入节流+`Scope.USER`+Checkpointer 白名单）；警告未 BLOCK | **持久化是 C1 必要条件**；遵 memory-rules（任务完成节点写、显式 user scope、不每步写）；Checkpointer breaking 须迁移方案 |
| **Q5-δ 分工** | 熟悉/信任/亲密是三独立构念、与 attitude(valence) 部分相关不等同；若只加一维**信任维正交性最强**；familiarity/exposure 是状态机**输入驱动信号**非独立关系维 | — | — | **attitude(v,a) 保持独立（情感评价）；C1 是独立关系层；初始态由 persona L3 `initial_attitude` 种子；若只加一维先加 trust（与 attitude 正交）** |

---

## 三、跨学科张力与收敛

本轮四席高度收敛，无根本分歧，张力均为"轻重缓急"非"对错"：

- **张力1 P4 紧迫性**：神经/数学均确认 A 是语义正确选择，但当前分布（峰 +0.278 未进危险区）使"现在上完整 A"非紧迫。收敛：廉价 cap 先落（防御兜底），完整 A 排次优先立项。
- **张力2 c 上界**：数学 [1.5,3.0] vs 神经 [1.5,2.5]。收敛取交集 **c∈[1.5,2.5]**（神经更保守、当前分布更安全）。
- **张力3 Q5-γ 持久化**：心理"必要条件" vs CS"成本真实"（未 BLOCK）。收敛：**持久化是必要条件**，成本由工程侧管理（迁移方案 + memory-rules 合规），不因成本降为可选。
- **张力4（已消解）C2 vs C1**：心理（构念无据）+ 数学（参数辨识差）两独立维度均否 C2。**C2 永久排除，不留可选项**。

---

## 四、决策

### P3 — 关闭确认（PASS）
`ATTITUDE_SETPOINT_A=0` + P1-c `arousal_baseline=-0.08` 已实现静息回归；实测 arousal 均值 -0.017≈0，神经席确认 gain=1.0 隐式吸收清醒 tonic。**P3 作 P1 附带修正正式关闭，不再单独跟踪。**

### P4（分期）
**当前批次可立即落地 — P4-d 廉价 cap**：`affect_core.py` 在 `arousal_gain` 后加 cap 钳制，env `ZERO_AROUSAL_GAIN_CAP`（默认 None=关，[0.3,0.6]）。`AROUSAL_GAIN` 常量不动、新参数经 state 注入（对齐 arousal_baseline）。**简化代价**：cap 仅截高端、不体现 a>0.5 自然反转，现分布从未触发、主要作未来防御。
**次优先立项 — P4-a 完整倒 U**：`affect_core.py:45` 改 `1 + c·prior_a·(1-prior_a)`，c∈[1.5,2.5] via `ZERO_AROUSAL_GAIN_C`；触发条件 arousal>0.5 实测（N=3/24 阈值，注释标注）。P4-c 负段弱衰减 d∈[0.1,0.3] 与 Q7 配套、单独立项。

### Q5（分期）
**当前批次可立即落地（须标局限）— A+B 止血**：
- **A**：复用已落地 `self.exposure` 派生 familiarity，`rate_eff=rate·(1-k·familiarity)`，env `ZERO_ATTITUDE_RATE_DECAY_K`（默认 0=关，零回归）。
- **B**：派生 `familiarity_label` 注入 `_CONVERSE_SYS` 作关系距离提示，env `ZERO_RELATIONSHIP_STAGE_HINT`（默认关）。
- **必须标局限**：A+B 仍是单不动点系统，只减缓漂移 + LLM 软约束，**不产生真多稳态**、**不宣称"Q5 已解决"**——定性"止血，C 立项"。对外文档不得描述为多稳态关系动力学。**简化代价**：A 的 familiarity 仅会话内（不跨会话）、重启归零；B 依赖 LLM 遵从软提示、非确定性。

**独立里程碑立项 — Q5-C1 离散关系状态机**：三阶段（陌生/初识/亲近，可扩 Knapp 细粒度）；计数器门控跃迁（确定性、禁 LLM 判）；事件：互惠自我披露/显著正事件/冲突降级；N_up 见数学席、N_down 2-5；首加信任维（trust，与 attitude 正交）；L3 `initial_attitude` 作种子；**跨会话持久化**（`Scope.USER`、任务完成节点写）。**前置条件**（工程实现前须完成）：(1) Checkpointer 迁移方案（新字段默认值兜底 + `runner.py:ALLOWED_CHECKPOINT_TYPES` 白名单）；(2) memory-rules 合规设计；(3) 完整 Q5-β 事件判据规格（防 LLM 判跃迁渗入）。**C2 pitchfork 永久排除，不得以"复用 mood_step 省事"为由引入**。

---

## 五、执行优先级

**本轮工程可立即落地（env 门控 + 零回归）**：

1. **P4-d 廉价 cap**：`ZERO_AROUSAL_GAIN_CAP` 默认 None 关，cap∈[0.3,0.6]。
2. **Q5-A**：`chat_driver` 复用 `self.exposure` 派生 familiarity 传 `rate_eff`；`ZERO_ATTITUDE_RATE_DECAY_K` 默认 0。
3. **Q5-B**：`familiarity_label` 注入 `_CONVERSE_SYS`；`ZERO_RELATIONSHIP_STAGE_HINT` 默认关。

**立项排后（须先满足前置）**：4. P4-a 完整倒 U（c∈[1.5,2.5]，触发 arousal>0.5） · 5. Q5-C1 离散状态机（Checkpointer 迁移 + memory-rules 合规） · 6. P4-c 负段弱衰减（与 Q7 联立）。

**关闭**：P3（静息基线）正式关闭 · C2 pitchfork 关系双稳永久排除。

---

## 六、红线自查

| 红线 | 状态 |
| --- | --- |
| 禁跨层反向依赖 | P4 cap/倒 U 纯 agents 层标量、无跨层调用 |
| 禁每条消息写记忆 | Q5-C1 持久化须任务完成节点写；A+B 无记忆层改动 |
| 禁默认 user scope | Q5-C1 显式 `Scope.USER`；A+B 无 scope 问题 |
| 禁 LLM 入 affect 热路径 | P4 纯标量；Q5 跃迁须确定性计数器（警戒线 LLM 判跃迁→BLOCK） |
| 议会只读不下场 | 只给设计范围与裁决，c/N_up/cap 均给范围不定单值 |

---

## 七、悬而未决

1. Q5-β 三类触发的检测实现细节（C1 立项时联合确认，防 LLM 渗入）。
2. `ZERO_AROUSAL_GAIN_MODE`（linear/cap/inverted_u）开关接口，工程师定规格、默认 linear 零回归。
3. Q5-C1 初始阶段与 `persona.initial_attitude` 映射规则。
4. 信任维（trust）操作化（什么事件增加 trust），C1 立项时定。
5. P4 负段弱衰减与 Q7 deactivation 联立出方案。

---

## 八、结论：PASS（分项）

| 项 | 结论 |
| --- | --- |
| P3 静息基线 | PASS → 关闭 |
| P4-d 廉价 cap | PASS → 可立即落地（env 门控） |
| P4-a 完整倒 U | PASS 设计语义 → 立项排后（当前分布未触发） |
| Q5-α C1 vs C2 | PASS C1，C2 永久排除 |
| Q5-A+B 先行 | PASS → 可立即落地，**须标"止血非已解决"局限** |
| Q5-C1 离散状态机 | PASS 设计语义 → 立项，待前置条件满足 |

---

## 引文（各席现场核验，去重汇总）

- Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function. *Annu. Rev. Neurosci.* 28:403-450. [DOI:10.1146/annurev.neuro.28.061604.135709](https://doi.org/10.1146/annurev.neuro.28.061604.135709) — LC-NE 倒 U 增益曲线，P4-a 神经依据。
- Berridge, C. W., & Spencer, R. C. (2016). Differential cognitive actions of NE at α2A and α1 receptors. *Brain Research* 1641(Pt B):189-196. [PMC4876052](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876052/) — α2A(高精度)/α1(低精度)双侧衰减分子机制，P4-c 依据。
- Arnsten, A. F. T. (2015). Stress signalling pathways that impair prefrontal cortex. *Nat. Rev. Neurosci.* [DOI:10.1038/nrn3896](https://doi.org/10.1038/nrn3896) · Inverted-U [Arnsten Lab](https://medicine.yale.edu/lab/arnsten/research/invertedu/) · PNAS 2025 [PMC12280923](https://pmc.ncbi.nlm.nih.gov/articles/PMC12280923/) — α1/α2 相反效应 + 倒 U 峰位归一化。
- Knapp, M. L. (1978). *Social Intercourse: From Greeting to Goodbye*. — 关系 10 阶段（升/降级不对称、需事件触发不自动升级）。[Wikipedia](https://en.wikipedia.org/wiki/Knapp%27s_relational_development_model) — Q5-α C1 依据、C2 排除构念依据。
- Altman, I., & Taylor, D. A. (1973). *Social Penetration*. — 亲密度离散层次可在任意阶段稳定/退化。[communicationstudies.com](https://www.communicationstudies.com/communication-theories/social-penetration-theory) — Q5-δ 三独立构念。
- Zajonc, R. B. (1968). Attitudinal effects of mere exposure. *JPSP Monograph*. [DOI:10.1037/h0025848](https://doi.org/10.1037/h0025848) — 曝光效应，Q5-A familiarity 依据。
- Fazio, R. H. (2007). Attitudes as object-evaluation associations. [PMC2677817](https://pmc.ncbi.nlm.nih.gov/articles/PMC2677817/) · [DOI:10.1521/soco.2007.25.5.603](https://doi.org/10.1521/soco.2007.25.5.603) — attitude 与关系维分工，Q5-δ。
- Reis, H. T., & Shaver, P. (1988). Intimacy as an interpersonal process. [PubMed 9599440](https://pubmed.ncbi.nlm.nih.gov/9599440/) — 亲密独立构念、需互动回应，Q5-γ 持久化依据。
- Gottman, J. M., & Murray, J. D. (2002). *The Mathematics of Marriage: Dynamic Nonlinear Models*. MIT Press. [MIT Press](https://direct.mit.edu/books/monograph/2547/The-Mathematics-of-MarriageDynamic-Nonlinear) · [ResearchGate 232424148](https://www.researchgate.net/publication/232424148) — 双稳是"稳定幸福/不幸"非"陌生/亲密"，C2 移植失真依据。
- Pitchfork bifurcation（超临界双稳条件 self_gain·self_k>1-inertia）[Wikipedia](https://en.wikipedia.org/wiki/Pitchfork_bifurcation) · Strogatz [作者页](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering) — C2 参数辨识性/双稳数学背景。
- Russell, J. A. (1980). A circumplex model of affect. *JPSP* 39(6):1161-1178. [DOI:10.1037/h0077714](https://doi.org/10.1037/h0077714) — (v,a) 环状空间，P3 静息 arousal≈0 参照。

---

*综合者：科学家议会主持。本纪要属议会只读裁决，不改 `src/`、不介入运行时数据产生。P4-d cap 与 Q5-A+B 可立即进工程队列，C1 须先满足前置条件。2026-07-01。*
