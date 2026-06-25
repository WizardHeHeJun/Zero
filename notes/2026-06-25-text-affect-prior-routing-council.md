# 科学家议会决策：文本→(v,a) 回归器接入时 AppraisalAgent 先验路由

- 日期：2026-06-25
- 议题：风险 3 —— 文本输入路径启用时，文本回归器产出的 (valence, arousal) 在内核里该接到哪一环（先验/证据/独立流）
- 评审席位：心理（评价理论）· 数学（贝叶斯）· 神经（预测编码）· CS（红线/可行性，必到）；主持席综合
- 治理：全程只读、强制现场引文、不下场改代码、不介入运行时数据产生。本文件是议会唯一的写动作（设计决策文档）。
- 背景产物：`feat/perception-text-affect` 分支已把 STTextAffectRegressor 接进 PerceptionAgent（默认关、零回归、已过 code-review）。本决策针对"再往内核走一步"的算法语义。

## 三个候选

- **(a)** 维持现状：文本只进 PerceptionAgent 的 `features`，调用方显式填 OCC，AppraisalAgent 仍 `occ_prior`。
- **(b)** 文本 (v,a) 替换/补充 `prior_mu`，绕过 `occ_prior`。
- **(c)** 两路独立，加权融合进内核后验。

## 各席判定汇总

| 席位 | (a) | (b) | (c) | 关键引用 |
| --- | --- | --- | --- | --- |
| 心理（评价理论） | 忠实 | **失真** | 简化（可防守） | OCC 1988; Scherer CPM 2001; Russell 1980/2003; Kuppens 2012 |
| 数学（贝叶斯） | 简化 | 简化（一致性最高） | **失真**（共享信源时） | Ernst & Banks 2002; Friston 2005/2009; Mihaylova 2017 |
| 神经（预测编码） | **失真** | 简化（层级正确） | **忠实**（须带精度权重） | Barrett 2017; Lindquist 2017; LeDoux & Brown 2017; Menon & Uddin 2010 |
| CS（红线/可行性） | 合规 | 合规（WARN：TD 语义分裂） | 合规（WARN：耦合成本高） | 现存 BUG: affect_math.py:296-297 + perception.py:77 |

## 跨学科张力的甄别与调和

### 张力 1：心理 ⟂ 神经 —— 文本 VA 是"评价下游读出"还是"高层语义 top-down prior"？
两席的 "prior" **所指不同**，是真实架构分歧而非纯术语错位：
- 心理席的 prior = OCC 式**结构化评价先验**（goal/standard/attitude 是评价的输入维度，驱动 `occ_prior`）。Kuppens 2012：appraisal 与 core affect 分开测量，appraisal 是**前因**，把已测 VA 反塞回 `occ_prior` 入口是因果倒置。
- 神经席的 prior = 预测编码层级中**高阶皮层（aMPFC/vlPFC）的概念性 top-down 预测**（Barrett 2017；Lindquist 2017：语言情感激活高阶皮层）。
- **收敛点**：现有代码只有一个 `prior_mu` 槽位（对应中阶评价 mPFC/ACC），它既不该被 measured VA 覆写（心理），文本语义又确实是更高阶的独立信号（神经）。两席共同指向——**文本 VA 不得作为 `occ_prior` 的输入维度，但可作为平行独立的高阶信号在更下游融合**。即受约束 (c)。

### 张力 2：数学 ⟂ 神经 —— (c) 的"失真"与"忠实"是否同一方案？
**可调和，指向同一约束**：
- 数学判 (c) 失真的前提 = "文本信源同时进独立流和 `occ_prior` 来源 + 朴素精度相加 → double counting，后验过度自信"（Mihaylova 2017）。
- 神经判 (c) 忠实的前提 = "文本流须带精度权重，不能固定权重平均"。
- 两前提不矛盾，合取即合法 (c)：**(i) 文本流信源独立（仅进独立流，不进 `occ_prior` 入口）；(ii) 精度加权融合（非朴素相加）**，等价 Ernst & Banks 2002 条件独立下的 MLE 最优估计。

### 共识（独立于 a/b/c）：现存 BUG
四席多数独立指出：`fast_survival_prior`（affect_math.py:296-297）按 OCC 布局假设 `features[0]=goal_congruence`、`features[3]=intensity`；而文本路径（perception.py:77）产出 `features=[valence, arousal, intensity, 0.0]`。`workspace_enabled=True` 时 valence 被当 goal_congruence、0.0 被当 intensity，**静默污染生存流**。默认 `workspace_enabled=False` 故休眠未暴露，但是真实缺陷，**必须先修，且独立于方案选择**。

## 决策

### 前置必做项（独立决策，先于 a/b/c，不可绕过）
修复 features 布局错位 BUG。建议路径（CS 席）：在 `Stimulus` 新增可选字段 `text_affect: tuple[float, float] | None = None`；`fast_survival_prior` 调用处按 backend 显式选用正确来源（或由调用节点显式传参），不再隐式依赖 `features` 下标布局。不动 AffectCore 核心逻辑，零回归风险最低。

### 方案决策：受约束的方案 (c)
两路独立 + 精度加权融合进内核后验，**不改 `occ_prior` 入口、不改 AppraisalAgent 签名**。
- **(b) 否决**：因果倒置（Kuppens 2012 appraisal→core affect 单向；Scherer CPM 2001 多维评价结构被抹平）+ TD 语义分裂（reward 仍来自 OCC goal_congruence，与文本 prior_mu 不一致，CS 席 WARN）。
- **(a) 否决**：神经席层级失真判定成立（语义信号=晚期高阶皮层 top-down，接进 early-perception/survival 特征是层级颠倒，违 Barrett 2017 / Lindquist 2017）；心理席的"(a) 忠实"依赖"文本不污染 survival"，但现存 BUG 表明 workspace 路径下已污染——"忠实"是休眠态幻觉。

### 受约束 (c) 落地必须满足的约束
1. 文本流**不进** `occ_prior` 入口（`AppraisalAgent.__call__` 不变，仍读 Stimulus 的 OCC 字段）。
2. 文本流**不进** `fast_survival_prior` 的 features（由前置 BUG 修复保证）。
3. 文本流以**独立 term** 进入 `fuse_terms`，精度**显式低于** `occ_prior` 产出精度（初版可类比 `SURVIVAL_PRECISION=0.4` 固定低值）。
4. 精度加权遵循条件独立假设（Ernst & Banks 2002），**不做朴素精度相加**导致 double counting。

### 可接受的刻意简化（及代价）
1. 文本流精度固定低值（非从模型不确定性动态估计）。代价：高质量文本时低估文本流权重，后验稍偏 OCC 先验。依据：Friston 2005 允许固定精度作初始近似。可后续从回归残差导出。
2. 文本流只贡献 (v,a) 标量，不引入 OCC 三维结构。代价：丢 Scherer CPM / OCC 多维评价信息（无法区分"道德违反"vs"目标阻碍"）。接受因文本→多维 OCC 暂无可靠自动实现。
3. 融合点在 AffectCore 的 `fuse_terms`，不改 AppraisalAgent / `occ_prior` 签名。代价：文本路径下 `prior_mu` 先为中性、最终 `post_mu` 才被文本修正，存在两阶段中间态。

## 悬而未决
1. 文本流精度如何动态估计（当前固定低值；未来从回归器残差/置信区间导出）。
2. 同一 stimulus 同时有 OCC 标注与 text 时，两路信源独立性如何形式化验证（目前靠不同字段输入假设独立，未测协方差结构）。
3. 受约束 (c) 落地后需专项回归测试：文本路径 + `workspace_enabled=True` 的端到端快照。

## 引文（各席现场核验）
- Ortony, Clore & Collins (1988). *The Cognitive Structure of Emotions*. Cambridge UP. [DOI:10.1017/CBO9780511571299](https://doi.org/10.1017/CBO9780511571299) — OCC 评价理论，多维结构化先验。
- Scherer, K. R. (2001). Appraisal as multilevel sequential checking. In *Appraisal Processes in Emotion*. Oxford UP. [Oxford Academic 章节](https://academic.oup.com/book/53557/chapter/422115725) — 评价是多阶段前因；VA 属下游 feeling 组件。
- Russell, J. A. (1980). A circumplex model of affect. *JPSP* 39(6):1161-1178. [DOI:10.1037/h0077714](https://doi.org/10.1037/h0077714).
- Russell, J. A. (2003). Core affect and the psychological construction of emotion. *Psych. Review* 110(1):145-172. [DOI:10.1037/0033-295X.110.1.145](https://doi.org/10.1037/0033-295X.110.1.145) — core affect "object-free, directed via appraisal"。
- Kuppens, P. et al. (2012). The dynamic interplay between appraisal and core affect in daily life. *Front. Psychol.* 3:380. [PMC3466066](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3466066/) — appraisal 与 core affect 分开测量，appraisal 是前因。
- Ernst, M. O. & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature* 415:429-433. [DOI:10.1038/415429a](https://doi.org/10.1038/415429a) — MLE cue integration 须条件独立。
- Friston, K. (2005). A theory of cortical responses. *Phil. Trans. R. Soc. B* 360:815-836. [DOI:10.1098/rstb.2005.1622](https://doi.org/10.1098/rstb.2005.1622) — 精度加权预测编码。
- Friston, K. (2009). The free-energy principle: a rough guide to the brain? *Trends Cogn. Sci.* 13:293-301. [DOI:10.1016/j.tics.2009.04.005](https://doi.org/10.1016/j.tics.2009.04.005).
- Mihaylova, L. et al. (2017). Distributed Multisensor Data Fusion under Unknown Correlation and Data Inconsistency. *Sensors* 17(11):2472. [PMC5713506](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5713506/) — 共享信源未建模 → 协方差低估（double counting）。
- Barrett, L. F. (2017). The theory of constructed emotion. *SCAN*. [PMC5390700](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390700/) — 概念知识 = 高阶皮层 top-down prior。
- Lindquist, K. A. et al. (2017). The role of language in emotion: a neuroimaging meta-analysis. *SCAN* 12(2):169-183. [PMC5390741](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5390741/) · [DOI:10.1093/scan/nsw121](https://doi.org/10.1093/scan/nsw121) — 语言情感激活 aMPFC/vlPFC 高阶皮层。
- LeDoux, J. & Brown, R. (2017). A higher-order theory of emotional consciousness. *PNAS*. [PMC5347624](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5347624/).
- Menon, V. & Uddin, L. Q. (2010). Saliency, switching, attention and control: a network model of insula function. [PMC2899886](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2899886/).

## 结论
**NEEDS-CHANGES（前置修 features 布局 BUG）→ 修后 PASS（受约束方案 c 可进 `/engineer` 落地）。**

四席在约束集下无 BLOCK：心理 ⟂ 神经张力在"文本流不进 `occ_prior` 入口"收敛；数学 ⟂ 神经张力在"信源独立 + 精度加权"收敛。

相关 file:line：
- `src/agents/affect_math.py:296-297`（BUG 所在：fast_survival_prior 的 features 下标假设）
- `src/agents/perception.py:77`（文本路径 features 布局）
- `src/orchestration/state.py`（Stimulus 新增 `text_affect` 可选字段）
- `src/agents/appraisal.py`（受约束 (c) 下不改签名）
- `src/agents/affect_core.py`（fuse_terms 接独立文本流的落点）
