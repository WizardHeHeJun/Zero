# 科学家议会·第二轮提案（待评审）：P4 arousal_gain 倒 U + Q5 关系多稳态语义裁决

> **议会评审输入（提案），不改 `src/`**。承接首轮 [seeking-attractor-council 纪要](2026-07-01-seeking-attractor-council.md)：P1（直流偏置根治）+ P2（习惯化）**已落地并实测**（阶段 31，`code-reviewer` PASS）。本轮处理议会当时排后的三项——**P3 关闭确认 · P4 arousal_gain 倒 U 语义 · P5(Q5) 关系多稳态语义裁决**——均属 [议会定语义]，工程师不能私自落地，故回议会。附工程师侧只读观测数据。

## 一、首轮落地实测（以项目结果为分析起点）

修复态旋钮（`ZERO_INTENSITY_FLOOR=0` · `ZERO_AROUSAL_BASELINE=-0.08` · `ZERO_ATTITUDE_REVERSION_A=0.4` · `ZERO_HABITUATION_TAU=7`），确定性端到端探针（真引擎、lm=None 词典评价、map 读出、固定种子）：

- **同一 24 轮日常友好对话**：落 seeking（暧昧/渴求）象限 **旧 6/24 → 修复 0/24 轮**；末轮 emotion arousal **旧 +0.177 → 修复 +0.032**。→ seeking 吸引盆在推荐态被拆除。
- **P3 观测（修复态 30 轮纯中性）**：arousal 均值 **−0.017**、峰 +0.122/谷 −0.105，**围绕静息 0 波动、无单调正漂移**（对照首轮旧态单调爬到 +0.06~0.12 正平台）。
- **P4 观测（修复态 24 轮强情绪）**：arousal 峰 **+0.278**、进入 arousal_gain 倒 U 危险区（>0.5）**0/24 轮**。

## 二、P3 裁决请求：Q3 静息基线——建议关闭

首轮纪要 §四 Q3 决策："Q1/Q2 修完后重测；若已趋近 0，Q3 视为 P1 附带修正、关闭；若仍有明显正偏置，提交议会定小正静息值。"
**工程师观测**：修复态静息 arousal 均值 −0.017（趋 0，无正偏置）。→ **建议议会确认 Q3 关闭**（`ATTITUDE_SETPOINT_A=0` + P1-c 的 `arousal_baseline` 已实现静息回归，无需再引独立正静息值）。请神经/生物席确认"静息 arousal≈0（而非小正值）"是否可接受，或坚持清醒 tonic 应为小正。

## 三、P4 裁决请求：arousal_gain 倒 U（`affect_core.py:45`）

首轮神经席判**失真**（现 `1+AROUSAL_GAIN·max(0,prior_a)` 线性无界，高唤醒段应倒 U 反转，Aston-Jones&Cohen / Arnsten α-1）；数学席判**非燃眉**（当前 arousal 低量级、不发散）。排 P4。请本轮定语义：

- **Q-P4-a 函数形式**：倒 U 取 `1 + c·a·(1−a)`？还是线性 + cap `min(1+c·a, 1+cap)`？还是分段？（数学席给稳定性、神经席给 LC-NE 曲线忠实度）
- **Q-P4-b 参数范围**：`c`（增益系数，现 AROUSAL_GAIN=1.0）与峰位/cap 的推荐**范围**（议会给范围、不定单值）。
- **Q-P4-c 负 arousal 段**：现 `max(0,·)` 使负 arousal（deactivation）gain=1；神经席指"低唤醒 LC 放电低、精度也应降"。是否让 gain 在负段 <1？
- **Q-P4-d 优先级**：观测显示当前分布强情绪也只到 +0.278、不进 >0.5 危险区 → 是否**先加廉价 cap 防御**（防未来真高唤醒场景），完整倒 U 待高唤醒实测出现再上？还是维持排后不动？

## 四、P5 裁决请求：Q5 关系多稳态语义（α/β/γ/δ）

工程师已出三路径工程方案（耦合估算 + 推荐）：[q5-relationship-multistability-engineering-plan.md](2026-07-01-q5-relationship-multistability-engineering-plan.md)（A: rate 随熟悉度衰减最轻 / B: familiarity 只作 LLM 距离标签不进热路径 / C: 离散态或 tanh 双稳最重）。请议会定语义：

- **Q5-α**：关系用**离散阶段**（C1 状态机）还是**连续双稳**（C2，复用本仓库 `mood_step` 的 pitchfork）？（心理席主裁 Knapp 阶梯 vs 连续；数学席给两者的不动点/盆宽/参数）
- **Q5-β**：跃迁的**确定性事件判据**是什么（正事件累计阈值？冲突降级？如何保证"时间不触发升级"）？——必须确定性、不得用 LLM 判（CS 席红线）。
- **Q5-γ**：关系态是否**必须跨会话持久化**（C 的记忆层耦合来源），还是会话内近似（路径 A）够？（CS 席给记忆耦合成本）
- **Q5-δ**：关系维度（熟悉/信任/亲密）与 P1-b 的 `attitude`(valence 长期评价) 是否**正交**、如何分工？（心理席建构效度）
- **工程师推荐（供裁）**：**A+B 先行**（低成本、正交、均零回归，作 P1/P2 后第二轮止血），**C 立项**待议会定 α/β/γ/δ 后实施；不与 A/B 并行赶工。请议会裁"A+B 是否可先落 / C 走哪条路径"。

## 五、红线自查

- 本提案只读项目结果（探针数据）+ `src/` 代码，工程师**未替引擎定运行时数值**（P4 的 c/cap、Q5 参数均请议会给**范围**）；未介入数据产生。
- P4/P5 候选修法均在"不把 LLM/meta 塞进 affect 热路径"内（P4 是纯标量函数、Q5 跃迁门控须确定性）。
- P3 观测是只读复现（内存后端、无 LLM），不改运行代码。

## 引文（复用首轮纪要现场核验条目，链接从略见首轮纪要 §引文）

- Aston-Jones & Cohen (2005) LC-NE 倒 U [DOI:10.1146/annurev.neuro.28.061604.135709](https://doi.org/10.1146/annurev.neuro.28.061604.135709) · Arnsten (2015) α1/α2 [PMC4876052](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876052/) —— P4。
- Knapp (1978) 关系阶梯 [Wikipedia](https://en.wikipedia.org/wiki/Knapp%27s_relational_development_model) · Gottman&Murray (2002) 婚姻数学双稳 [MIT Press](https://direct.mit.edu/books/monograph/2547/The-Mathematics-of-MarriageDynamic-Nonlinear) · Strogatz 非线性动力学 pitchfork —— Q5。

---
*第二轮提案，2026-07-01。P1/P2 已落地实测；本轮请议会裁 P3 关闭 / P4 倒 U 语义 / Q5 α-δ，裁决后另立纪要、工程师据以落地。*
