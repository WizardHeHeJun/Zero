# 科学家议会决策：情绪「防抖」旋钮默认值 + 采样失真根因（2026-06-30）

> 触发：`--chat` 实测「逐轮情绪标签翻号」（用户连续敌意时某轮 e\* 被采样成强正 valence，标签从「愤怒」跳「兴奋」、与对话内容解耦）。已落地两个 env 旋钮（默认=旧常量、零回归）：`ZERO_SAMPLE_SIGMA_MAX`(默认0.5)、`ZERO_EMOTION_NOISE_STD`(默认0.05)。本次议会评议：**默认值取多少 + 翻号根因 + 采样 vs MAP**。
> 议会属性：只读顾问、强制现场核验文献、**不介入情绪/记忆/语言数据的产生**（见 [[analysis-results-first-no-intervene]] 与 `notes/2026-06-25-science-council-design.md`）。
> 路由 4 席：数学 / 心理 / 神经 / CS（生物席与本议题弱相关，控成本略过）。

## 结论：**NEEDS-CHANGES**

本次提案（给两旋钮拍新默认值）被判为「**有效的临时安全网，但不是根治**」。四席一致：在上游先验未修复时，降 cap 对**典型**敌意输入几乎无效（真实 `post_sigma≈0.175` 已低于 cap=0.5），只有降到 0.10–0.12 才对当前态有缓解效力。真正的失真在**先验量级**与**逐轮 i.i.d. 采样违反情绪时序结构**。

## 各席判定汇总

| 席位 | 判定 | 推荐 `SIGMA_MAX` | 推荐 `NOISE_STD` | 核心依据 |
| --- | --- | --- | --- | --- |
| 数学 | **失真**（逐轮 i.i.d. 读出、违时序自相关） | 0.10 | 0.01 或 0（改 AR(1)/OU） | 单样本读出不对应任何标准损失；MCMC/OU/AR(1) 时序相关 |
| 心理 | **失真**（构念效度失败；上游标定太软） | 0.15–0.20 | 0.01–0.02 | 翻号=情境-情绪解耦=病态；健康 AR1≈0.20–0.33 |
| 神经 | cap=0.5 **失真**；i.i.d. **简化**；单样本 **简化** | 0.12 | 0.02 | 群体编码平均压缩；内感受高层精度；皮层 MCMC |
| CS（守红线） | **NEEDS-CHANGES**（无 BLOCK、无红线违规） | 不改代码默认，走 `.env.example` | 同左；eval 设 0 | cap 对典型输入不生效；根因=`standard_compliance=0` |

## 跨学科张力（主持收敛）

### 张力 1：翻号根因的分层定位——两个诊断不互斥、有上下游关系
- **CS / 数学**：根因在 `standard_compliance` 恒 0 → `occ_prior` 的 0.3 权重被结构性置零、量级稀释约 40%（敌意句 valence 仅 ≈-0.35）。
- **心理**：`occ_prior` 公式（0.5/0.3/0.2）**没错**；问题在更上游 `lm.appraise_text` 标定太软——敌意按 OCC anger 触发条件应给 `goal_congruence∈[-0.7,-0.9]`，却只给 -0.2~-0.3。
- **收敛**：二者**叠加**。即便 appraise 标定改准，standard_compliance 仍恒 0 则 OCC anger 的「规范违背」通道缺失、fuse 仍偏弱；反之亦然。需**两路并行**，不是二选一。

### 张力 2：`SIGMA_MAX` 推荐值离散度（0.10 / 0.12 / 0.15–0.20）
- 数学·神经按「真实后验 sigma（典型敌意 ≈0.175）」定值——cap 须低于它才生效 → **0.10–0.12 是当前态唯一有实效的区间**。
- 心理允许 0.15–0.20 是**上游修复后**的目标态（先验变强、后验本身更确定，可放宽探索）。
- **收敛**：现状（上游未修）取 **0.10–0.12**；上游修好后可放宽到 0.15–0.20。

### 张力 3：`NOISE_STD` 额外噪声的定性
- 当前态（无 AR(1)/OU 结构）下，叠加在无记忆读出上的额外 gauss 噪声**只放大失真** → 降到 **0.01–0.02**，**eval 必须设 0**。
- 该噪声当前用全局 `random.gauss`、**不受 `rng_seed` 控制 → 不可复现**（CS 席独立指出的工程缺陷）。

### 张力 4：改 OCC `standard` 维度语义的权限边界（共识，未被挑战）
- 把 `standard_compliance` 从恒 0 **接上现有 appraisal 信号** = 工程接线，**工程师团队可做**。
- **重定义** OCC standard 维度的心理语义（它测量什么、如何与愤怒联系）= **需回议会**（心理+CS 联合）。工程师不得私自定语义。

## 决策正文

### A. 临时安全网（议会推荐值 → 写 `.env.example`，**不改代码默认**）
- `ZERO_SAMPLE_SIGMA_MAX`：推荐 **0.10–0.12**（当前上游未修态下有实效；上游修好后可放宽 0.15–0.20）。
- `ZERO_EMOTION_NOISE_STD`：推荐 **0.01–0.02**；**eval 环境必须设 0**；该噪声应接入 `rng_seed` 以保可复现。
- CS 席治理原则：**不得**在代码里给 `ZERO_SAMPLE_SIGMA_MAX` 硬编码非空默认（否则测试/eval 环境悄然漂移）；代码默认维持 `None→MAX_SAMPLE_SIGMA`，推荐值只走文档。

### B. 可接受的刻意简化（及代价）
- i.i.d. 采样（非 MCMC 序列相关）：临时可接受，代价=轨迹缺时序相关；AR(1)/OU 实现前以降 cap 缓解。
- 单样本读出（非多采样均值）：临时可接受，代价=读出噪声偏高；神经席建议 N=5–10 均值为**研究方向**，不作当前强制。
- 日志输出离散情绪词：可接受，代价=丢后验分布信息；不改。

### C. 必须改 / 根治路径（按优先级 + 归属）
1. **P1（结构性失真·工程师可做接线）**：`standard_compliance` 从恒 0 接上现有 appraisal 信号 → `chat_driver.py:143-148` Stimulus 构造。
2. **P2（需回议会）**：若 P1 涉及**重定义** OCC standard 心理语义 → 心理+CS 联合评审后再实现。
3. **P3（上游标定·需回议会）**：`lm.appraise_text` 的 `goal_congruence` 标定校准（敌意应给 -0.7~-0.9）→ 属 appraisal 学术准则，心理席评审后给工程指导。
4. **P4（读出结构·设计需回议会、工程师实现）**：逐轮 i.i.d. → AR(1)/OU 时序相关，或多采样取均值（N=5–10）。读出公式由数学+神经确认后实现。
5. **P5（可复现性·工程师可做）**：`random.gauss` 噪声接入 `rng_seed`。

### D. 悬而未决
- 多采样均值（N=5–10）的算力/对话延迟代价：未评估，需工程师给延迟测算。
- AR(1) ρ 取值（心理 0.20–0.33 实证 lag-1；数学 0.3–0.6）：来源不同，需数学+心理联合确认单值。
- `lm.appraise_text` 能否仅靠 prompt/标定文件改写而不改模型：技术路径未明，工程师先探查。

## 红线自查
- 议会全程**只读**项目结果与 `src/` 代码，强制现场核验文献，**未介入情绪/记忆/语言数据的产生**。
- 仅给出「设计参数范围 + 根因分析」（属开发期架构评审，非替引擎跑/选数据）；代码默认值不变，推荐值走 `.env.example` 文档，由主程/工程师团队执行写入。
- 「给推荐默认区间」vs「替引擎定输出情绪」的边界：本次属前者（合规）；任何「让议会参与运行时生成」的产物不采纳。

## 引文（各席现场核验，遵 `.claude/rules/meeting-notes-citations.md`）

- Buesing, L. et al. (2011). Neural dynamics as sampling. *PLoS Comput. Biol.* 7(11):e1002211. [DOI:10.1371/journal.pcbi.1002211](https://doi.org/10.1371/journal.pcbi.1002211) · [PMC3207943](https://pmc.ncbi.nlm.nih.gov/articles/PMC3207943/) — 神经采样=MCMC 序列相关、需 burn-in，非 i.i.d.。
- Oravecz, Z. et al. (2011). A hierarchical latent stochastic differential equation model for affective dynamics. *Psychol. Methods* 16(4):468-490. [PubMed:21823796](https://pubmed.ncbi.nlm.nih.gov/21823796/) · [PDF](https://ppw.kuleuven.be/okp/_pdf/Oravecz2011AHLSD.pdf) — 核心情感动态服从 OU 过程、随机涨落时序相关。
- Kuppens, P. et al. (2010). Emotional inertia and psychological maladjustment. *Psychol. Sci.* 21(7):984-991. [DOI:10.1177/0956797610372634](https://doi.org/10.1177/0956797610372634) · [PMC2901421](https://pmc.ncbi.nlm.nih.gov/articles/PMC2901421/) — 健康情绪 AR(1) ρ≈0.20-0.33；过高惯性=适应不良。
- Houben, M. et al. (2015). The relation between short-term emotion dynamics and psychological well-being: a meta-analysis. *Psychol. Bull.* 141(4):901-930. [PubMed:25822133](https://pubmed.ncbi.nlm.nih.gov/25822133/) — 变异性/不稳定性/惯性三构念 meta；MSSD 正常≈0.07 vs MDD≈0.40。
- Rottenberg, J. et al. (2005). Emotion context insensitivity in MDD. *J. Abnorm. Psychol.* 114(4):627-639. [PMC3624976](https://pmc.ncbi.nlm.nih.gov/articles/PMC3624976/) — 情绪与情境解耦=病态标志。
- Kuppens, P. & Verduyn, P. (2017). Emotion dynamics. *Curr. Opin. Psychol.* 17:22-26. [DOI:10.1016/j.copsyc.2017.06.004](https://doi.org/10.1016/j.copsyc.2017.06.004) · [PDF](https://ppw.kuleuven.be/okp/_pdf/Kuppens2017ED.pdf) — 变异性/不稳定性/惯性辨析。
- Hollenstein, T. (2007). State space grids. *Int. J. Behav. Dev.* 31(4):384-396. [DOI:10.1177/0165025407077765](https://journals.sagepub.com/doi/10.1177/0165025407077765) — lability/rigidity 两端皆调节失败。
- Ortony, Clore & Collins (1988). *The Cognitive Structure of Emotions*. Cambridge UP. [DOI:10.1017/CBO9780511571299](https://doi.org/10.1017/CBO9780511571299) — OCC anger 触发条件；`standard_compliance` 语义来源。
- Kuppens, P. et al. (2012). The dynamic interplay between appraisal and core affect. *Front. Psychol.* 3:380. [PMC3466066](https://pmc.ncbi.nlm.nih.gov/articles/PMC3466066/) — appraisal 是 core affect 前因；标定软→先验弱。
- Barrett, L. F. & Simmons, W. K. (2015). Interoceptive predictions in the brain. *Nat. Rev. Neurosci.* 16(7):419-429. [DOI:10.1038/nrn3950](https://doi.org/10.1038/nrn3950) · [PMC4731102](https://pmc.ncbi.nlm.nih.gov/articles/PMC4731102/) — EPIC：高层内感受后验精度高、sigma 远小于 0.5。
- Haefner, R. M. et al. (2016). Perceptual decision-making as probabilistic inference by neural sampling. *Neuron* 90(3):649-660. [DOI:10.1016/j.neuron.2016.03.020](https://doi.org/10.1016/j.neuron.2016.03.020) — 皮层 MCMC 序列相关；i.i.d.=简化。
- The Autocorrelated Bayesian Sampler. [PMC11115360](https://pmc.ncbi.nlm.nih.gov/articles/PMC11115360/) — 人类概率判断 MCMC 序列自相关、1/f 噪声，否定 i.i.d.。
- Barrett, L. F. (2017). The theory of constructed emotion. *Soc. Cogn. Affect. Neurosci.* 12(1):1-23. [PMC5390700](https://pmc.ncbi.nlm.nih.gov/articles/PMC5390700/) — 群体编码期望；单样本≠群体均值。
- Ernst, M. O. & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature* 415:429-433. [DOI:10.1038/415429a](https://doi.org/10.1038/415429a) — 精度加权融合；后验均值为最优读出基准。
- Fiser, J. et al. (2010). Statistically optimal perception and learning. *Trends Cogn. Sci.* 14(3):119-130. [PMC2939867](https://pmc.ncbi.nlm.nih.gov/articles/PMC2939867/) — 采样假说（用于感觉皮层、非情绪输出层）。
- Hoyer, P. & Hyvärinen, A. (2002). Interpreting neural response variability as Monte Carlo sampling of the posterior. *NIPS 2002*. [NeurIPS](https://proceedings.neurips.cc/paper/2002/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html) — 神经变异性作后验采样的奠基。
- Bayes estimator. *Wikipedia*. [link](https://en.wikipedia.org/wiki/Bayes_estimator) — 二次损失→后验均值（MMSE）；0-1 损失→MAP；单样本=次优读出。
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *J. Basic Eng.* 82(1):35-45. [DOI:10.1115/1.3662552](https://doi.org/10.1115/1.3662552) — 最优状态估计基准（CS 席：prior mean 偏弱时压方差不能拉离分界）。

---

## 二轮议决（先验量级修复 P1/P3，心理 + CS 两席，2026-06-30）

把一轮路由回议会的「先验 valence 太弱」议决成**可实现决定**。两席在 P1(a) 上正面分歧、由心理席（OCC 语义权威）裁定。

### 各席判定

| 提案 | 心理席（OCC 语义） | CS 席（红线/可行性） | **议会决定** |
| --- | --- | --- | --- |
| **P1(a)** `standard_compliance = goal_congruence` | **失真·不得实施**——合并 OCC 正交两维=删除 reproach 通道 | 条件 PASS，但语义裁决权让给心理席 | ❌ **否决**（心理席失真裁定优先） |
| **P1(b)** appraise 增独立 standard 维度 | **忠实·议会背书**（主路径），须 eval 查与 valence 共线性 | NEEDS-CHANGES：协议 breaking，优先新增 `appraise_full()` 不改现签名 | ✅ 忠实主路径，**排中期**（P3 之后） |
| **P1(c)** 确定性信号派生 | 简化·可作轻量回退 | — | 备选回退 |
| **P3** `_APPRAISE_SYS` 分级标定校准 | **PASS**，给分级校准锚（须分级、防误伤"语气直但中性"） | **PASS**，加 `ZERO_APPRAISE_CALIBRATE` env 门控、独立 PR | ✅ **可做**，先行 |
| **P5** rng_seed 接管 noise | — | PASS（已完成） | ✅ 已落地（commit `ec15f26`） |

### 关键裁决与依据

- **P1(a) 被否**：OCC（[OCC 1988](https://doi.org/10.1017/CBO9780511571299)）中 anger = **distress（goals 维）+ reproach（standards 维）** 复合，两维正交。令 standard=goal 即删 reproach 通道，丢 Lazarus「a demeaning offense against me and mine」的 standards 特质（[Core relational theme](https://en.wikipedia.org/wiki/Core_relational_theme)）。**用户言语攻击时 standard_compliance 应独立取强负**（粗鲁 -0.5~-0.7，明确侮辱/谩骂 -0.8~-1.0），与 goal_congruence 各自贡献。
- **P3 先行**：只改 `_APPRAISE_SYS` 提示词常量（单文件、低风险）；P1(b) 依赖 P3 标定先定。LLM 有系统性正向偏置（positivity bias，[arXiv:2507.21083](https://arxiv.org/abs/2507.21083)：GPT-4 对负向输入给负向响应概率仅中性输入的 1/3），正是 `appraise_text` 对敌意只给 -0.2~-0.3 的成因。
- **P3 必须分级**（心理席给的校准锚，按语义距离插值、非硬规则）：中性「你好」≈0 · 轻微批评「这回答不太对」≈-0.2 · 明确不满「总让我失望」≈-0.4 · 明确敌意「你真没用」≈-0.75 · 极端谩骂≈-0.95。一刀切「负面→-0.7」会误伤"语气直但中性"输入。
- **P3 红线边界**（CS）：改 `_APPRAISE_SYS` = 「校准仪器量程」非「替仪器读数」→ 属「议会定准则、工程师建机制」合规分工，**不违**「议会不下场生成」。须加 `ZERO_APPRAISE_CALIBRATE` env 门控保零回归（影响 chat + 研究 `generate` 的 VAD 反推两路）。

### 执行顺序（议会定）

1. **P3**（env 门控分级校准）——工程师先做，两席 PASS。
2. **P1(b)**（独立 standard 维度）——P3 标定稳定后做；新增 `appraise_full()`、配共线性 eval；忠实主路径。
3. **P1(a)** 永久否决；**P1(c)** 仅作 LLM 不可用时的轻量回退。

### 红线自查
两席全程只读、未介入数据产生；仅给构念忠实裁定 + 标定准则 + 归属/门控建议。P3 的标定锚是「准则」非「替某句话定数值」，合规。

### 二轮新增引文（现场核验）
- Lazarus, R. S. (1991). *Emotion and Adaptation*. OUP. — anger 核心关系主题「a demeaning offense against me and mine」。[Core relational theme (Wikipedia)](https://en.wikipedia.org/wiki/Core_relational_theme) · [Appraisal theory (Wikipedia)](https://en.wikipedia.org/wiki/Appraisal_theory)
- Smith, C. A. & Lazarus, R. S. (1993). Appraisal components, core relational themes, and the emotions. *Cognition & Emotion* 7(3-4):233-269. [ResearchGate](https://www.researchgate.net/publication/247497157_Appraisal_Components_Core_Relational_Themes_and_the_Emotions) — anger=goal incongruence 高 + other-blame 高。
- Steunebrink, B. et al. (2009). The OCC model revisited. [PDF](https://people.idsia.ch/~steunebrink/Publications/KI09_OCC_revisited.pdf) — OCC 形式化，确认 goals/standards 维正交。
- Khorshidifar, F. et al. (2025). ChatGPT Reads Your Tone and Responds Accordingly — Until It Does Not. [arXiv:2507.21083](https://arxiv.org/abs/2507.21083) — LLM 正向偏置：负向输入响应被压向中性，解释 appraise 标定太软。
- Anger intensity continuum. [Anger (Wikipedia)](https://en.wikipedia.org/wiki/Anger) — annoyance→rage 连续体，支持分级标定。

---

## 三轮议决（P4 读出结构，数学 + 神经两席，2026-06-30）

把一轮的「逐轮 i.i.d. 采样违情绪时序自相关」议决成**可实现读出设计**。两席**完全收敛**于 **α（MAP 读出）**。

### 决定：`e* = post_mu`（MAP/MMSE 读出），env 门控、默认旧行为 → **已实现**

| 候选 | 数学席 | 神经席 | 决定 |
| --- | --- | --- | --- |
| **α** MAP `e*=post_mu` + 既有 AR1≈0.4 | **忠实·推荐** | **忠实·推荐** | ✅ **采纳·已实现** |
| β e\* 上叠显式 AR(1)/OU 噪声 | 简化→失真（与 AR1 双重计数 → ARMA(2,1) 不可识别） | 简化（冗余） | ❌ |
| γ 跨轮携带 rng 做 MCMC-like | 失真（延续伪随机流 ≠ Metropolis 链，仍 i.i.d.） | 失真（对话 5-30s ≫ 皮层采样相关 10-350ms，生物学无对应） | ❌ |
| δ 每轮多采样 N=5-10 取均值 | 简化（α 的有损近似，N→∞ 即 α） | 简化（可选，留作表达层可变性） | ❌（核心读出） |

### 关键依据

- **α + 既有 AR1 = OU 一阶正确离散化**（数学席）：离散等间隔 OU 等价 AR(1)，`ρ=e^{−θΔt}`。`emotion_decay_step` 的 `recovery=0.4` 即 AR(1) ρ=0.4，被确定性外力 `post_mu` 驱动 → 干净三层分解：**惯性←AR1 · 信号←post_mu · 涨落←独立 `ZERO_EMOTION_NOISE_STD`**。post_mu 已是精度加权后的 MMSE 最优点估计（[MMSE](https://en.wikipedia.org/wiki/Minimum_mean_square_error_estimator) · [Bayes estimator](https://en.wikipedia.org/wiki/Bayes_estimator)），再采样只增方差不增信息。
- **生物学尺度错配**（神经席）：皮层 MCMC 采样去相关 10-350ms（[Murray 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4241138/) · [Hennequin 2014](https://arxiv.org/abs/1404.3521)），对话轮 5-30s 是其 15-3000 倍——跨轮采样链早已混合，γ 无生物对应；逐轮"错"不在采样相关而在**单样本大方差 + 跨轮无惯性**，后者已由 AR1≈0.4 修。GNW ignition 后是**稳定代表性广播读出**（[Mashour 2020](https://www.cell.com/neuron/fulltext/S0896-6273(20)30052-0)），支持 post_mu（期望）而非单样本；下游读出取期望而非单粒子（[Orbán 2016](https://www.sciencedirect.com/science/article/pii/S0896627316306390)）。
- **AR1≈0.4 实证合法**：负性情绪 lag-1 自相关实测 0.33-0.40（[Kuppens 2015 PMC4705270](https://pmc.ncbi.nlm.nih.gov/articles/PMC4705270/)），`EMOTION_RECOVERY=0.4` 正落其中，等效时间常数 τ≈1.1 轮（每轮 10-20s → 11-22s），吻合快变情绪衰减窗。

### 实现（本轮已落地）
- 落点 `affect_core.py:83`：`if state.affect_readout=="map": e_star=post_mu else sample_affect(...)`。
- 控制字段 `AffectState.affect_readout: str="sample"`，照搬 `sample_sigma_cap`/`rng_seed` 经 runner→state 穿透；`chat_driver` 读 `ZERO_AFFECT_READOUT` 注入。
- **`emotion_decay_step` 不动**（AR1≈0.4 是正确时序机制）；**`sample_affect` 保留**——神经席留作表达层「同一 e\* → 可变表达」简并性（**follow-up**：表达层可变性与情绪读出解耦，本轮未拆）。
- 默认 `sample` → 逐字旧行为，零回归。回归 369 passed。

### 三轮新增引文（现场核验）
- Murray, J. D. et al. (2014). A hierarchy of intrinsic timescales across primate cortex. *Nat. Neurosci.* 17(12):1661-1663. [PMC4241138](https://pmc.ncbi.nlm.nih.gov/articles/PMC4241138/) — 皮层内禀自相关 50-350ms。
- Hennequin, G. et al. (2014). Fast sampling-based inference in balanced neuronal networks. *NeurIPS*. [arXiv:1404.3521](https://arxiv.org/abs/1404.3521) — 优化网络采样去相关≈10ms。
- Mashour, G. A. et al. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron* 105(5):776-798. [Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(20)30052-0) — GNW ignition 后稳定再入广播。
- Kuppens, P. et al. (2015). Emotional inertia... *PMC4705270*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4705270/) — 负性情绪 AR1≈0.33-0.40 实证。
- Minimum mean square error estimator. [Wikipedia](https://en.wikipedia.org/wiki/Minimum_mean_square_error_estimator) — MMSE = 后验均值。
- Elliptical Ornstein–Uhlenbeck process. [arXiv:2001.05965](https://arxiv.org/pdf/2001.05965) — OU 均匀采样等价 AR(1) 的离散化推导。
