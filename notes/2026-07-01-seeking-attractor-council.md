# 科学家议会综合纪要：seeking 吸引盆——多轮对话情绪滑向暧昧的确定性根因与裁决

**评审对象**：`notes/2026-07-01-arousal-baseline-dc-bias-council-proposal.md` — 情感网络 seeking 吸引盆（--chat 多轮对话情绪读数平滑却 ~20 轮后系统性滑向暧昧；九环同向耦合；Q1–Q7 + arousal_gain + seeking 象限 + TD）

**评审时间**：2026-07-01 ｜ **席位**：数学 / 心理 / 生物 / 神经 / CS（守红线）｜ **评审状态**：**NEEDS-CHANGES**（多项 NEEDS-CHANGES，无 BLOCK）

> 治理：议会只读、强制现场核验文献、不下场生成、不介入运行时数据产生。本纪要只给设计决策与推荐**范围**，不替引擎定运行时单值。提案（评审输入）见 [2026-07-01-arousal-baseline-dc-bias-council-proposal.md](2026-07-01-arousal-baseline-dc-bias-council-proposal.md)。

---

## 一、触发与现场（机制精确化）

用户 12 轮真实 --chat 对话中，情绪读数 `emotion_label` 全程在「欣喜/专注」往复，未翻负；attitude 的 arousal 维单调从 +0.07 爬至 +0.11 平台并锁定；约 20 轮后对话进入约饭→定时间→"我该怎么认出你"→"迟到就打电话"的暧昧升温序列。用户判断：这与当下对话内容不符，是引擎层面的系统性漂移。

**机制精确化（数学席核心贡献）**：这不是 Gottman 式非线性双稳分叉（`mood_step` 的 pitchfork 那类），而是**线性系统的单稳不动点被直流偏置推到正 arousal 区**。`attitude_step` 已有 reversion 项（2026-06-29 必改已落），能正确把不动点钳在 `a* = rate·ē_a / (rate + reversion)` 而非令其无限漂移——reversion 的数学功能是正确的；真正的失真在**输入信号本身是整流的恒正直流**，reversion 只能把不动点拉向 setpoint，拉不过恒正的输入流。修法策略随之明确：**根治直流偏置只需消除整流输入或把 setpoint 设成负值抵消（Q1/Q2），不必在 attitude 动力学里引入非线性**（非线性只在 Q5 关系多稳态那条独立的更深需求里才需要）。

"~20 轮"恰是两个时间尺度的交汇：attitude 累积平台约 τ ≈ 1/rate = 12–22 轮（rate=0.08, reversion=0.01），记忆窗 40 entries ≈ 20 轮（`chat_driver.py:201-204`），后者挤出"初次见面"等距离锚点（primacy_k=5 仅覆盖约 2.5 轮，不足以长期维持"我们才刚认识"的上下文）。两者同时成熟：引擎持续向 LLM 注入"欣喜·seeking"心情信号，同时 LLM 的工作记忆中"陌生人"语境已消失，LLM 自身的 rapport 升级先验接手 → 对话不可避免地单调升温。

---

## 二、各席判定汇总表

> 符号说明：忠 = 忠实；简 = 可接受简化；简★ = 简化已到失真边界；失 = 失真·必改

| 议题 | 数学席 | 心理席 | 生物席 | 神经席 | CS 席 | 综合判定 |
|---|---|---|---|---|---|---|
| **Q1** arousal 整流直流底噪（`abs` + `intensity` 下限 0.2） | 失 | 失 | 失（非生理静息） | 失（正段方向忠实，下限人工制品） | — | **失·必改** |
| **Q2** attitude 在 arousal 维做慢累积 | 失（语义无据） | 失·必改主裁（Fazio/Scherer/Frijda） | 倾向"不累积 arousal 维"，让心理席主裁 | — | 建议 Q2(b) 零 breaking | **失·必改**（走 Q2(b)） |
| **Q3** arousal 中性点/静息基线 | 简偏失（单调上移无习惯化=失真） | 简偏失（Kuppens 2013 V 形：中性 valence → 低非恒正 arousal） | 失（显式静息基线缺失；单调上移=allostatic load） | 忠（清醒 tonic LC 放电非零→小正平台可接受），但无习惯化=失真 | — | **简偏失·NEEDS-CHANGES**（静息值争议存在，见张力；习惯化缺失是独立失真=Q6） |
| **Q4** emotion 基线混合把 attitude_a 平台带入情绪基线 | — | — | — | — | 依赖 Q2 缓解（工程接线） | **条件性改善**（Q2 修后自动部分缓解；`w=0.6` 混合本身已部分缓解，余量视 Q2 结果） |
| **Q5** 关系多稳态（标量 EWMA vs 离散跃迁） | 失（标量单稳只有唯一不动点；调参无解） | 失（Knapp/Altman-Taylor：依恋=事件门控离散跃迁，EWMA 连"陌生态"稳态都没有） | 简★（关系阶段跃迁真实，但不是最紧迫失真） | 简★（TD 无跨轮学习=关闭习惯化通道，架构约束下可接受） | 不建议直上多状态机（C 路径）；建议先给工程方案 | **失·NEEDS-CHANGES**（深层，建议先出工程方案，不与 Q1-Q2 并行改） |
| **Q6** 缺习惯化/hedonic adaptation | 失（重复刺激应衰减） | 失·必改（2026-06-29"可选"翻案） | 失·必改（SCR 习惯化 5-10 次趋零；每次对同一对象全新反应=失真） | 失·必改（Schultz DA-RPE：完全可预期奖励 → 响应回零；日常友好应习惯化） | — | **失·必改**（跨席一致翻案；优先级高） |
| **Q7** arousal 双向性（平淡对话是否主动降唤醒） | 失（abs 消灭 deactivation 臂，只剩上半圆） | — | 失（副交感 vagal brake 主动降唤醒是真实机制） | 失（负 arousal 段 `max(0,·)` 截断=deactivation 通路被堵） | — | **失·必改** |
| **arousal_gain** 唤醒对精度的增益（`affect_core.py:45`） | 非燃眉（不致不稳定，只加速收敛） | — | — | 失（高唤醒段应倒 U 反转，Aston-Jones&Cohen；现线性=高唤醒正反馈无界） | 可门控 env 变量 | **失·但非燃眉**（排直流偏置之后处理；倒 U 优先级 P4） |
| **seeking 象限标签** `motivational_system(v+,a+)→"seeking"` | — | 标签粗近似可接受，失真在 LLM 层 | 失（SEEKING/CARE 混淆=不同神经内分泌系统） | 失（seeking=通用探索非亲密专有；Berridge wanting≠liking≠CARE） | — | **裁定见下文张力§** |
| **TD 无跨轮学习**（key=`user_text[:40]`） | — | — | — | 简（架构约束下可接受，但关闭习惯化通道） | 简（架构约束） | **简·可接受**（架构约束下不改，习惯化由 Q6 独立处理） |

---

## 三、跨学科张力与收敛

### 张力一：seeking 象限——引擎贴标失真（生物/神经席）vs 标签粗近似、失真在 LLM 层（心理/神经席内部）

**生物席依据**：`motivational_system(v+,a+)→"seeking"` 混淆 Panksepp SEEKING（多巴胺/探索/wanting，非亲密专有）与 CARE（催产素·内啡肽/bonding/依恋），二者神经内分泌底物不同，不应共用同一标签。

**心理/神经席依据（标签侧）**：标签只是粗近似（Berridge 已区分 wanting/liking/learning 三义），孤立的 `seeking` 贴标并不必然导致暧昧；**真正失真在下游 LLM 层**——`_CONVERSE_SYS` 把"欣喜·seeking"直接传给 LLM，LLM 读到"seeking"结合"rapport 上升"背景，自身的 rapport 升级先验将其解读为亲密许可。

**收敛裁定**：两方均正确，但层次不同。

- 引擎侧贴标：`seeking` 作为粗近似标签在 `v+,a+` 象限是公认的描述性近似（Panksepp 原著确实把该象限归 SEEKING），**本身不是燃眉失真**。引擎层面不强制要求把 `seeking` 拆成 `seeking_explore/care_bonding` 两个子标签（改标签不解决 LLM 解读问题，且标签拆分的边界本身尚无共识）。
- LLM 层的语义漂移是**独立失真**：把"欣喜·seeking"翻译成暧昧许可的是 LLM 提示层，而非引擎贴标本身。修法归属在 LLM 提示侧（`_CONVERSE_SYS` 措辞、人设卡增加"保持分寸"约束）——此属语言生成层，**不在本议会职权内下场定措辞**，但**确认这是失真**，建议工程团队（语言模块）修订提示逻辑，并由议会在专题提案中定"如何区分 CARE vs SEEKING"的语义边界。
- **悬而未决**：`motivational_system` 是否在 `v+, a+` 内部按 arousal 阈值（如 arousal > 0.5 且 v > 0.5 → care，否则 → seeking）进一步分层，留下一轮议会专题（需 Panksepp 原著 arousal 阈值实证）。

### 张力二：Q2(a) 去 arousal 维 vs Q2(b) 独立 reversion_a/setpoint_a

**语义侧（心理席）**：attitude 是对象指向的 valence 维评价（Fazio attitude-object 联结；Scherer 把 sentiment 定义为 valence 倾向性评价；Frijda：attitude = disposition to emotional response，核心是 valence）。"对某人的长期唤醒基线"在心理学文献中无先例，属语义失真。偏好 Q2(a)：attitude 只累积 valence。

**工程侧（CS 席）**：attitude 是已持久化的二元组 `(v, a)`，Q2(a) 去掉 arousal 维会破坏序列化结构。Q2(b)（独立 `reversion_a`，加大到足以抵消直流偏置；独立 `setpoint_a`）零 breaking——不改数据结构，不改接口。

**收敛裁定**：采 **Q2(b)**。

理由：(1) 心理席的语义判定是正确的——attitude_a 不应单调累积；但"去掉"和"让它快速回到静息"在行为上等价，且前者有 breaking 风险；(2) Q2(b) 通过设 `reversion_a` 足够大使得 `a* ≈ setpoint_a`（即稳态 arousal ≈ 0），在功能上实现"attitude 不累积 arousal 偏置"，同时保留数据结构向后兼容；(3) 代价：attitude_a 在技术上仍存在，语义上是"快速均值回归至静息的 arousal 感受性"，若未来要彻底去除仍需升版数据格式——记为悬而未决。实施路径：在 `.env.example` 暴露 `ZERO_ATTITUDE_REVERSION_A`（默认值远大于 `reversion_v=0.01`，具体值由议会给推荐量级，见下文）和 `ZERO_ATTITUDE_SETPOINT_A`（默认 0.0）。

**推荐量级（可接受的简化边界）**：`reversion_a` 需满足 `rate·ē_a / (rate + reversion_a) ≈ 0`，即 `reversion_a >> rate·ē_a`。已知 `rate=0.08`，`ē_a` 在中性对话约 0.077，偏正对话约 0.15–0.20。要使稳态 `a* < 0.02`（不可感知），需 `reversion_a ≈ 0.3–0.5`（比现有 `reversion=0.01` 大 30-50 倍）。此量级是工程参数，**议会给推荐范围 [0.3, 0.5]，工程师走 env 暴露，不由议会定运行时单值**。

### 张力三：Q6 从 2026-06-29"可选"翻案为"必改"——依据是否足够？

**2026-06-29 原判**："习惯化递减"列为"未做·可选"，属探索改进。

**翻案依据（四席一致）**：

- 生物席：SCR（皮肤电导）习惯化在 5–10 次重复非威胁刺激后趋零（Groves & Thompson 双过程理论）；社交接触中对同一对象每次都给出全新的高唤醒反应，违反最稳健的自主神经事实。
- 心理席：Frederick & Loewenstein 享乐适应；Kuppens 2010/2013 情感惯性研究；重复友好互动在正常个体中会降低情感强度。
- 神经席：Schultz DA-RPE：完全可预期的奖励刺激中 RPE 趋零（无意外 = 无 DA 响应），当前 TD 用 `user_text[:40]` 做 key 使每轮都是新 key、永远不更新同一条目 = 人为制造永恒新奇 = 习惯化通道被堵。
- 数学席：缺习惯化是吸引盆结构的组成部分（九环中的第 9 环），无对抗正向漂移的衰减。

**裁定**：翻案依据充分，Q6 升级为**失真·必改**。原"可选"判定是在没有完整动力系统分析的情况下做出的局部评估；本次九环体系级诊断显示习惯化缺失是结构性失真，不是优化项。翻案。

### 张力四：arousal_gain 倒 U 优先级

**神经席**：现有线性增益 `arousal_gain = 1 + AROUSAL_GAIN * max(0, prior_a)`（`affect_core.py:45`）在高唤醒段变为正反馈（唤醒越高 → 精度越大 → 下轮唤醒更锁定），应改为倒 U（Aston-Jones-Cohen 曲线：`1 + c·a·(1-a)` 或加 cap），低-中段方向忠实。

**数学席**：当前 arousal 基线在 0.1 量级（非接近 1 的极端），该范围内正反馈效应虽存在但不是燃眉失真，不致发散——只加速向直流偏置不动点的收敛。

**生物/CS 席**：同意数学席，排直流偏置之后。

**收敛裁定**：倒 U 是正确的长期方向（失真），但**不是燃眉优先级**——在直流偏置（Q1/Q2）仍将 arousal 钉在 0.1 量级的情况下，倒 U 修正效果被偏置噪声淹没；先去偏置、再在真实 arousal 分布上验证倒 U 的必要性，排 P4。归属 [议会定语义]（倒 U 函数形式和 cap 值）+ [工程接线]（替换一行表达式）。

---

## 四、决策：NEEDS-CHANGES

**整体判定**：NEEDS-CHANGES。无 BLOCK。引擎存在多项失真（Q1/Q2/Q6/Q7 是确定性的失真，Q3/Q5 是简化到边界），需按优先级逐步修正。

### 每项裁决

**Q1 | arousal 整流直流底噪**
- 裁决：**失真·必改**
- 决策：(1) `intensity` 下限 0.2 改为 0（或改为 env 可调旋钮 `ZERO_INTENSITY_FLOOR`，默认 0）；(2) 中性无事件输入时 arousal 证据应可回落到 0。`0.6·|valence|` 项（circumplex V 形关系）保留——该项有文献依据（Kuppens 2013）。
- 归属：下限值 [工程接线]，默认推荐值 0 由 [议会裁定]（见上）。
- 代价：去除下限后，完全中性输入的 arousal 证据降至 0，情绪在中性对话中将真正回落到静息——这正是目标。

**Q2 | attitude 在 arousal 维慢累积**
- 裁决：**失真·必改（语义）；走 Q2(b) 零 breaking 实现**
- 决策：`attitude_step` 不改接口，但 arousal 维的 `reversion` 参数独立化（`reversion_a` ∈ [0.3, 0.5]，via env `ZERO_ATTITUDE_REVERSION_A`），`setpoint_a` 独立化（默认 0.0，via env `ZERO_ATTITUDE_SETPOINT_A`）。语义上：引擎不再把"对此人的长期唤醒偏置"当作 attitude 的一部分建模。
- 归属：参数范围推荐 [议会裁定]；`attitude_step` 接口与 env 暴露 [工程接线]。

**Q3 | arousal 中性点/静息基线**
- 裁决：**简偏失·NEEDS-CHANGES（部分）**
- 决策：(1) Q1/Q2 修完后重新测量静息值——神经席"清醒 tonic LC 放电→小正平台"与生物席"显式静息基线"之间的争议，在直流偏置去除后的真实分布上再判；(2) 单调上移无习惯化的失真由 Q6 独立处理（习惯化递减覆盖了"随时间单调升"这条失真路径）；(3) 暂不引入独立 arousal setpoint（`ATTITUDE_SETPOINT_A=0` 已部分起到该作用）。
- 悬而未决：清醒静息 arousal 的合适正偏置量（小正 vs 真正中性 0），待 Q1/Q2 修后观测真实分布再决。

**Q4 | emotion 基线混合**
- 裁决：**条件性改善**（依赖 Q2）
- 决策：Q2(b) 修完后 attitude_a 稳态趋近 0，`baseline_a = w·attitude_a + (1-w)·setpoint_a ≈ 0`，自然缓解。`w=0.6` 参数（env `ZERO_EMOTION_BASELINE_ATTITUDE_W`）保留可调性。不独立改 baseline 公式。
- 归属：[工程接线]，跟随 Q2 联动。

**Q5 | 关系多稳态（标量 EWMA vs 离散跃迁）**
- 裁决：**失真，但非燃眉；先出工程方案再落**
- 决策：数学席"标量单稳调参无解"与心理席"陌生态本身需是稳态"均成立。当前实现是架构上的刻意简化已到失真边界。修法路径：CS 席建议 A（rate 随熟悉度衰减，仅 2 文件），先仅暴露 `familiarity_counter` 变量（不进热路径），由熟悉度门控 `rate` 衰减（高熟悉度 → 低 rate → 难以再升级），近似实现"陌生态更稳定"。完整多吸引子状态机（CS 席 C 路径）留后续，当前不并行实施。
- 归属：工程方案 [工程接线 + 议会审语义]；完整多稳态语义 [议会定语义，下轮提案]。

**Q6 | 习惯化/hedonic adaptation**
- 裁决：**失真·必改（2026-06-29"可选"翻案）**
- 决策：引入 exposure 计数机制。具体：在 `attitude_step` 或独立的 `habituation_step` 中，按同一对话对象的累计轮次对 arousal 输入施加衰减因子 `η(n) = exp(-n/τ_hab)`（`τ_hab` 为习惯化时间常数，初值参考 SCR 习惯化 5–10 次趋零 → `τ_hab ≈ 5–10` 轮，via env `ZERO_HABITUATION_TAU`）。衰减只作用于 arousal 分量（valence 的习惯化机制有争议，暂不引入）。
- 归属：`τ_hab` 推荐范围 [议会裁定]（5–10 轮，来自 Groves & Thompson / Schultz）；`habituation_step` 实现 [工程接线]。

**Q7 | arousal 双向性（deactivation 通路）**
- 裁决：**失真·必改**
- 决策：`occ_prior` 中 arousal 已是全 abs 整流，`evidence_from_value` 中 `abs(delta)` 同样为正。修法：(1) 在 Q1 修（`abs(intensity)` 改不整流强度）的基础上，允许"低强度、低 |valence|"的平淡输入给出负/零 arousal 证据（deactivation 信号）；(2) `attitude_step` 的 `setpoint_a=0` + 强 `reversion_a` 已实现"平淡时 attitude_a 向 0 回归"；(3) 在 `emotion_decay_step` 侧，若连续 N 轮 arousal 输入低于阈值，baseline_a 主动向负偏置（relaxation/deactivation 态）——此条较激进，先做 (1)(2)、观测效果后决定是否需要 (3)。
- 归属：(1)(2) [工程接线]；(3) 如需引入负偏置，语义需 [议会定语义]（副交感 vagal brake 量级）。

**arousal_gain 倒 U**
- 裁决：**失真·但非燃眉；P4**
- 决策：待 Q1/Q2/Q6 修完、观测真实 arousal 分布后，若高唤醒段正反馈效应仍明显，在 `affect_core.py:45` 将 `arousal_gain = 1 + AROUSAL_GAIN * max(0, prior_a)` 改为 `1 + c * prior_a * (1 - max(0, prior_a))` 或加 cap（`min(arousal_gain, 1 + AROUSAL_GAIN)`）。c 值和 cap 由 [议会定语义]（Aston-Jones-Cohen 倒 U 函数参数）；改一行表达式 [工程接线]。

**seeking 象限语义**
- 裁决：**引擎贴标是粗近似（可接受）；LLM 层语义漂移是失真（待专题提案）**
- 决策：`motivational_system` 的 `seeking` 贴标暂不修改。专题提案：在"亲密度积累"议题中定义 SEEKING vs CARE 的 arousal/valence 子象限边界，连同 `_CONVERSE_SYS` 措辞修订一起提交议会下轮评审。此专题不属于本次评审范围，由主代理跟踪。

**TD 无跨轮学习**
- 裁决：**简·可接受（架构约束）**
- 决策：维持。习惯化缺失由 Q6 独立的 exposure 计数处理，不在 TD key 结构上动手术。

---

## 五、执行优先级与顺序

以下按紧迫度与依赖关系排序：

**P1（燃眉·直流偏置根治，工程师现在可动）**

- P1-a：Q1 — 去除 `intensity` 下限（`chat_driver.py:156` `max(0.2, ...)` 改为 env 可调 `ZERO_INTENSITY_FLOOR`，默认 0）。归属 [工程接线]。
- P1-b：Q2(b) — `attitude_step` 暴露独立 `reversion_a`、`setpoint_a` 参数，走 env `ZERO_ATTITUDE_REVERSION_A`（推荐 0.3–0.5）和 `ZERO_ATTITUDE_SETPOINT_A`（默认 0.0）。归属 [工程接线]（语义裁定已由本纪要给出）。
- P1-c：Q7(1)(2) — `occ_prior` arousal 公式允许输入信号为零/负（去整流），配合 Q1 联动；`attitude_step` 的 `setpoint_a + reversion_a` 提供 deactivation 回归力。归属 [工程接线]（联动 P1-a/P1-b）。

依赖：P1-a/P1-b/P1-c 可并行，不互相 breaking，优先本轮实施。

**P2（必改·习惯化，工程师可动，议会已给量级）**

- Q6 — 引入 `habituation_step`，exposure 计数，arousal 输入衰减因子 `η(n) = exp(-n/τ_hab)`，`τ_hab` ∈ [5, 10]，via env `ZERO_HABITUATION_TAU`。归属 [工程接线]。需新增 exposure 计数状态，确认不进 affect 热路径的 LLM 分支。

**P3（必改·静息基线验证，P1 修完后）**

- Q3 — P1 修完后重新跑 30 轮中性对话，观测静息 arousal 分布；若仍有明显正偏置，提交议会定"合适的小正静息值"专题；若已趋近 0，Q3 视为 P1 的附带修正，关闭。此项**需先完成 P1 才能判断是否需要独立处理**。

**P4（失真但非燃眉·arousal_gain 倒 U，P1 观测后）**

- arousal_gain — P1/P2 修完、观测真实 arousal 分布后，若高唤醒正反馈仍显著，再修 `affect_core.py:45`。当前不动。需 [议会定语义]（倒 U 函数形式 + 参数推荐，提专题）。

**P5（深层结构性改造，先出工程方案）**

- Q5 — CS 席 A 路径（familiarity counter + rate 衰减）工程方案由工程师提出，提交议会审语义后实施；完整多稳态状态机留后续议会定语义。此项**工程师现在不可直接动**，需先出方案。
- seeking vs CARE 分层 — 专题提案，议会下轮审。

**待定（P1 后重新评估）**：Q4 随 Q2 联动改善，无需独立操作。Q7(3)（主动负偏置 deactivation）待 P1 后观测决定。

---

## 六、红线自查

- **只读**：综合以项目 trace 数据（提案 §一）和 `src/` 代码读取为起点，未介入情绪/记忆/语言数据的产生。所有判定以现有实现代码（`affect_math.py`/`affect_core.py`/`chat_driver.py`/`emotion_lexicon.py`/`value.py`）为分析对象，未修改任何文件。
- **不下场生成**：未替引擎定任何运行时数值（如具体的 `ZERO_ATTITUDE_REVERSION_A=0.35`）；给出的是推荐**范围**（[0.3, 0.5]、[5, 10] 轮）供工程团队在 env 实现后实验验证；倒 U 函数形式和参数未预设具体值。
- **不介入热路径**：所有候选修法均是确定性数学项修改（整流去除、参数独立化、exposure 计数）或 env 旋钮，无 LLM/meta 进入 affect 热路径。seeking/CARE 的下游 LLM 提示修订，明确标注"不在本议会职权内直接定措辞"。
- **CS BLOCK 尊重**：无 BLOCK 项；CS 席的工程建议（Q2(b) 零 breaking、Q5 先工程方案、不直上多状态机）已全部纳入决策。

---

## 七、可接受的刻意简化（备查）

| 简化 | 代价 | 接受依据 |
|---|---|---|
| TD key = `user_text[:40]`（无跨轮学习） | 关闭了 RPE 层面的习惯化通道；每轮等价新刺激 | 架构约束下可接受；习惯化由 Q6 独立 exposure 机制补偿 |
| `motivational_system` seeking 贴标（粗近似） | SEEKING/CARE 语义混淆，LLM 可能误读为亲密许可 | 引擎层贴标粗近似在文献中有依据；精化留专题 |
| `w=0.6` 混合基线（非纯 attitude） | 60% attitude 权重仍会部分传入直流偏置 | Q2(b) 修完后 attitude_a ≈ 0，此权重失去影响；参数保留灵活性 |
| Q5 暂用 familiarity_counter + rate 衰减而非完整多稳态 | 陌生态不是真正的吸引子，只是起始点，仍可能缓慢漂移 | 完整多稳态工程成本高，先以近似方案验证效果 |

---

## 八、落库与执行指针

- 本纪要为议会综合决策存档；评审输入（提案）见 [2026-07-01-arousal-baseline-dc-bias-council-proposal.md](2026-07-01-arousal-baseline-dc-bias-council-proposal.md)。
- 执行计划：P1-a/P1-b/P1-c 裁决可直接转 `/engineer` 工程任务（语义已由本纪要给定，工程师建机制、走 env 零回归）；P5（Q5 工程方案）需工程师先提草案再回议会。
- 后续实现遵红线：所有 env 旋钮默认=旧行为（零回归）、代码不硬编码非空默认（推荐值走 `.env.example`，同 2026-06-30 治理原则）；实现者≠审查者，落地后过 `code-reviewer` 独立门。

---

## 引文（各席现场核验，去重汇编；遵 `.claude/rules/meeting-notes-citations.md`）

**心理学·态度与情感结构**

- Fazio, R. H. (2007). Attitudes as object-evaluation associations of varying strength. [PMC2677817](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2677817/) — attitude = 对象-评价联结（valence 维为核心）。
- Scherer, K. R. (2005). What are emotions? And how can they be measured? *Social Science Information* 44(4):693-727. [DOI:10.1177/0539018405058216](https://doi.org/10.1177/0539018405058216) — sentiment = valence 倾向性评价，非 arousal 累积。
- Russell, J. A. (1980). A circumplex model of affect. *JPSP* 39(6):1161-1178. [DOI:10.1037/h0077714](https://doi.org/10.1037/h0077714) — valence×arousal 二维环状结构；arousal 双极性（deactivation↔activation）。
- Russell, J. A. (2003). Core affect and the psychological construction of emotion. *Psychological Review* 110(1):145-172. [DOI:10.1037/0033-295X.110.1.145](https://doi.org/10.1037/0033-295X.110.1.145) — core affect 个体基线/affective homeostasis；arousal 应有静息锚。

**情感动力学·惯性与吸引子**

- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psychological Science* 21(7):984-991. [DOI:10.1177/0956797610372634](https://doi.org/10.1177/0956797610372634) · [PMC2901421](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2901421/) — 高情感惯性=适应不良；attitude 无习惯化→病理类比。
- Kuppens, P., Tuerlinckx, F., Russell, J. A., & Barrett, L. F. (2013). The relation between valence and arousal in subjective experience. *Psychological Bulletin* 139(4):917-940. [DOI:10.1037/a0030811](https://doi.org/10.1037/a0030811) · [PubMed 23231533](https://pubmed.ncbi.nlm.nih.gov/23231533/) — V 形/boomerang：中性 valence → 低 arousal（非恒正）。
- Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical latent stochastic differential equation model for affective dynamics. *Psychological Methods* 16(4):468-490. [DOI:10.1037/a0024375](https://doi.org/10.1037/a0024375) — 情感时间序列的 OU 过程建模；均值回归参数估计。
- Gottman, J. M., & Murray, J. D. et al. (2002). *The Mathematics of Marriage: Dynamic Nonlinear Models*. MIT Press. [MIT Press](https://direct.mit.edu/books/monograph/2547/The-Mathematics-of-MarriageDynamic-Nonlinear) · [ResearchGate 232424148](https://www.researchgate.net/publication/232424148) — 关系动力学双稳不动点（非线性 influence 函数，非 EWMA）。
- Vallacher, R. R., & Nowak, A. (1994). *Dynamical Systems in Social Psychology*. Academic Press. [research page](https://psy2.fau.edu/~vallacher/research_DSP.html) — 社会心理动力系统；多稳态关系模型。
- Strogatz, S. H. *Nonlinear Dynamics and Chaos*. [作者页](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering) — pitchfork 分叉、单稳 vs 双稳临界条件、不动点稳定性判据。

**关系发展·阶段模型**

- Knapp, M. L. (1978). *Social Intercourse: From Greeting to Goodbye*. — 关系发展离散阶段（coming together/apart），非连续标量。[Wikipedia: Knapp's relational development model](https://en.wikipedia.org/wiki/Knapp%27s_relational_development_model)
- Altman, I., & Taylor, D. A. (1973). *Social Penetration: The Development of Interpersonal Relationships*. — 亲密度离散层次，事件门控而非时间函数。[communicationstudies.com](https://www.communicationstudies.com/communication-theories/social-penetration-theory)

**习惯化与享乐适应**

- Groves, P. M., & Thompson, R. F. (1970). Habituation: a dual-process theory. *Psychological Review* 77(5):419-450. [DOI:10.1037/h0029810](https://doi.org/10.1037/h0029810) · [ResearchGate 18847090](https://www.researchgate.net/publication/18847090) — 重复非威胁刺激应衰减；双过程理论。
- Frederick, S., & Loewenstein, G. (1999). Hedonic adaptation. In D. Kahneman et al. (Eds.), *Well-Being*. [条目](https://stafforini.com/works/frederick-1999-hedonic-adaptation/) — 持续刺激下情感强度递减。

**神经科学·Panksepp SEEKING / CARE**

- Panksepp, J. (1998). *Affective Neuroscience*. OUP. 综述 Panksepp (2018). [DOI:10.3389/fnins.2018.01025](https://doi.org/10.3389/fnins.2018.01025) · [PMC8406748](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8406748/) — SEEKING（DA/wanting/探索）与 CARE（催产素·阿片肽/bonding/依恋）的神经内分泌区分。
- Berridge, K. C., & Kringelbach, M. L. (2015). Pleasure systems in the brain. *Neuron* 86(3):646-664. [PDF](https://www.kringelbach.org/papers/Neuron_BerridgeKringelbach2015.pdf) — wanting vs liking vs learning 三义区分；SEEKING 非亲密专有。

**神经科学·NE 增益 / 唤醒调制**

- Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annual Review of Neuroscience* 28:403-450. [DOI:10.1146/annurev.neuro.28.061604.135709](https://doi.org/10.1146/annurev.neuro.28.061604.135709) — LC-NE 唤醒增益倒 U 函数；高唤醒段认知功能下降。
- Arnsten, A. F. T. et al. (2015). NE α2/α1 receptor signaling in the prefrontal cortex. *Brain Research*. [PMC4876052](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4876052/) — α-2 增益上升 vs α-1 应激损害；倒 U 机制。
- Aston-Jones et al. The LC-NE system in stress and arousal. [PMC7873441](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7873441/) — 清醒态 tonic LC 放电最高（静息唤醒基线非零）。

**神经科学·DA-RPE / 习惯化**

- Schultz, W. (2016). Dopamine reward prediction-error coding. *Dialogues in Clinical Neuroscience* 18(1):23-32. [PMC4826767](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4826767/) · [DOI:10.31887/DCNS.2016.18.1/wschultz](https://doi.org/10.31887/DCNS.2016.18.1/wschultz) — 完全可预期奖励 RPE 趋零；习惯化神经机制。

**自主神经 / 生理**

- Porges, S. W. (2009). The polyvagal theory. [DOI:10.3949/ccjm.76.s2.17](https://doi.org/10.3949/ccjm.76.s2.17) · [PMC3108032](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3108032/) — 副交感 vagal brake 主动降唤醒机制。
- Berntson, G. G., Cacioppo, J. T., & Quigley, K. S. (1991). Autonomic determinism. *Psychological Review* 98(4):459-487. [DOI:10.1037/0033-295X.98.4.459](https://doi.org/10.1037/0033-295X.98.4.459) — 自主神经二维双极模型；降唤醒=副交感正激活。
- Appelhans, B. M., & Luecken, L. J. (2006). Heart rate variability as an index of regulated emotional responding. *Review of General Psychology* 10(3):229-240. [DOI:10.1037/1089-2680.10.3.229](https://doi.org/10.1037/1089-2680.10.3.229) — HRV 作为主动情绪下调能力指标。
- Chen, K.-H. et al. (2014). Habituation of parasympathetic-mediated heart rate responses to recurring acoustic startle. *Front. Psychol.* 5:1288. [DOI:10.3389/fpsyg.2014.01288](https://doi.org/10.3389/fpsyg.2014.01288) · [PMC4238409](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4238409/) — 副交感恢复成分对重复刺激快速习惯化（心血管层面）。
- Goldstein, D. S., & Kopin, I. J. (2007). Evolution of concepts of stress. *Stress* 10(2):109-120. [PMC4166604](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4166604/) — allostatic load：持续激活使调节点偏离基线=病理。

---

*本纪要由科学家议会主持综合（数学/心理/生物/神经/CS 五席现场核验），2026-07-01。各席文献已现场核验，链接取自各席核验结果汇编，未臆造 URL。议会只读、不下场生成、不介入运行时数据产生；落库执行由主程完成。*
