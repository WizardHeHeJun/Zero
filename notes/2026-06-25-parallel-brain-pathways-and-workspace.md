# 并行脑路的生物学可行性 + 显著度门控全局工作空间框架

> **日期**：2026-06-25
> **性质**：一次神经科学文献调研的成果固化（承接 `2026-06-23-emotion-math-and-llm-expression.md` 总论、`2026-06-24-text-output-emotion.md` 文本通道；本篇收窄到**架构形态**：情感处理是否应/能并行，以及整合机制怎么建）。
> **缘起**：`diagrams/2026-06-23T010000` 画了"Supervisor → 6 脑区并行 → Integrator(全局工作空间·VAD 融合) → Expression"，但 `src/orchestration/graph.py` 的真实实现是一条**线性串行管线**。本篇论证图与代码的分叉该往哪边收，并据此给出改良框架（已落地为 `workspace_enabled` 门控的加法，见第五节）。
> **生成方式**：基于 WebSearch / WebFetch 检索的同行评审文献（30 篇，见末节）；核对说明同前两篇。

---

## 一、问题：并行脑路可行吗？

诊断图承诺"6 个脑区 = 6 个功能模块、齐刷刷并行涌入一个 Integrator"。要判断这是否可建、值不值得建，得回答三件事：(1) 大脑的情感处理是不是并行的？(2) 是不是"脑区 = 功能模块"那种并行？(3) 多路并行最后怎么整合成一个情绪状态？文献对三问的答案分别是：**是、不是、靠显著度门控的 ignition 广播**。

---

## 二、并行是真的——四种有硬证据的并行形式

**① 通往杏仁核不止一条路，且不止两条。** LeDoux 的"低路（丘脑→杏仁核，快粗）/高路（皮层→杏仁核，慢细）"双通路 [1] 在人脑有实证升级：基于 622 人 HCP 的弥散纤维追踪重建出**上丘→丘脑枕→杏仁核**皮层下捷径，纤维密度高者恐惧面孔识别更强 [3]；fMRI 证实该通路编码负性情绪强度 [4]，掩蔽恐惧面孔可绕皮层快速激活杏仁核 [6]。**关键修正**：Pessoa & Adolphs 2010 把"低路"重写为 **"many roads"**——皮层内大量短路捷径同样快，"快=皮层下专属"是错的 [2][7]。结论：**多条并行通路、速度/分辨率各异**成立，但 >2 条，且快慢≠皮层下/皮层。

**② 大尺度网络层面的并行 + 动态切换。** Menon 三网络模型（显著网络 SN / 默认网络 DMN / 额顶执行网络 FPN）是三套并行大网络，由**前岛叶+dACC 的显著网络做门控/切换** [20][21]。情绪不定位于单一脑区，而是**跨分布式、领域通用网络**实现 [11][12]。

**③ 并行的预测编码流。** 内感受（EPIC：边缘皮层下行内脏运动预测、上行内脏感觉，沿岛叶前后轴整合，服务 allostasis）[15][16] 与外感受是**并行的分层消息传递流**，按精度加权融合（自由能/主动推断框架，见旧笔记 B 节）。

**④ 价值信号的并行广播。** 多巴胺 RPE 由 VTA/SNc **以同步并行波广播**到整个额叶-基底节环路 [23][24]——"并行广播"在生物学里最干净的例子，正是 `value.py` 的 TD 原型。

---

## 三、三条反驳——为什么"1:1 脑区并行"是错的

**① 无模块化、无指纹。** Pessoa 双重竞争模型与《The Cognitive-Emotional Brain》：脑区多功能，"情感/认知"无法干净映射到分立脑块 [8]。Barrett 建构情绪论用**简并性/群体编码**说明"恐惧"由多变神经群的多种时空模式产生，**无单一神经指纹** [9][10]；分类器找到的模式只是统计摘要 [13]。所以**把"杏仁核=恐惧 agent"硬编码成 1:1 模块在解剖学上是假的**——这是图 010000 的核心毛病。

**② 评价不是全并行，是"并行组件 + 序列累积的检查"。** Scherer 成分过程模型(CPM)有 5 个并行功能网络（诱发/表达/自主/行动倾向/感受），**但** Grandjean & Scherer 的 EEG 证据显示评价**检查本身是序列的**：新异性→内在愉悦度→目标一致性依次展开、累积 [14]。纯并行错，真实是"流并行、流内检查序列、且有递归"。

**③ "Integrator"那个框藏起了最难的部分。** 把多路简单平均成一个 VAD 不是大脑做的事。生物学整合 = **全局神经工作空间的 ignition**：信息要赢得竞争、跨阈值才**非线性全或无地"点燃"额顶网络并广播**，否则停留局部（无意识）[17][18][19]；跨流绑定靠 **communication-through-coherence / γ 同步** [22]。整合器必须是**显著度门控 + 赢者通吃 + 再入广播**，而非被动求和。

**对快通路的重要降级（New LeDoux）**：杏仁核快路是**生存回路/动机状态**，**不等于"感到恐惧"**；情绪感受是更高阶、需全局广播的再表征 [25][26][27]。所以快通路应产**亚符号的生存/唤醒信号**，不直接喊"我恐惧"。

---

## 四、可行性裁决

并行架构**比线性管线更忠于生物学**——但前提是按三原则重画，否则朴素版反而更不科学：
1. 并行的是**功能流（时间尺度 × 精度）**，不是解剖脑区；
2. 整合是**显著度门控、精度加权、带 ignition 阈值的竞争**，不是平均；
3. 必带**自上而下再入**与**简并/多对一**，不硬编码脑区=情绪。

---

## 五、改良框架：并行预测编码流 + 显著度门控全局工作空间（已落地）

把线性链 + 图 010000 一起替换为三层。**默认关（`workspace_enabled`）、零回归的加法**，复用既有 `fuse_terms`（已能按精度融合任意 N 项）。

### Layer 1 — 并行流，每条吐 `(μ, Π)`，按时间尺度×精度分工
| 流（生物学锚点） | 角色 | 精度 Π |
| --- | --- | --- |
| **快生存流**（上丘-枕-杏仁核 [3] / 生存回路 [25]） | 原始特征出粗效价/高唤醒，最快、**低精度**，可单独点燃（威胁显著） | 低、固定 |
| **评价流**（OCC/CPM [14]） | 多维细评先验，精度随确定性升 | 随确定性 |
| **价值流**（VTA/RPE [23]） | TD-RPE 证据，并行项非串行前置 | π·唤醒增益 |
| **心境流**（A.7，`mood_enabled` 时） | 慢变双稳先验 | `MOOD_PRECISION` |

唤醒增益（NE/精度调制 [15]）：唤醒越高，评价/价值流精度（投票权）越大——把"内感受/身体激活"以精度而非独立 μ 的形式并入，避免手搓伪 μ。

### Layer 2 — 显著度门控全局工作空间（真 Integrator）
- **显著度门控竞争**：salience = 精度加权的偏离中性幅度（= SN 门控分数 [20]）；
- **ignition 阈值**：仅过阈流点燃进入全局 e*，亚阈流停留局部（对应生存回路无"感受"地作用 [25]）；无流过阈则保留最显著者（不空播）；输出"哪些流点燃"= 可解释；
- **精度加权再入**：把固定 `RECONCILE_WEIGHT=0.5` 换成 Kalman 式 `precision_reconcile`——高精度内核抗语言拉拢、低精度内核让步，affect↔language 回路成为真变分再入。

### Layer 3 — 简并感知表达
不硬编码脑区→通道；保留采样（`sample_affect` 天然实现"同一 e*→可变表达"的多对一 [9]）。

### 相对图 010000 的三处改正
1. 6 脑区 → 4 条功能流（按时间尺度/精度分，非解剖）；
2. 平均 Integrator → 精度门控 + ignition + 再入工作空间；
3. 快路从"恐惧 agent" → 亚符号生存/唤醒信号。

---

## 六、三道鸿沟（承接旧笔记，未跨越）
以上全是对**情感功能投影**的更忠实并行化：效度、坍缩、qualia 依旧未解。点燃广播让信息"全局可用"≠"有感受"——这正是 New LeDoux 与高阶理论 [25] 把"生存回路"和"情绪意识"分开的原因。并行化改进的是**信息流组织**，不触碰"被体验到"本身。

---

## 七、完整文献来源（30）

**并行通路 / 双通路争议**
1. LeDoux, *The Emotional Brain* / low-road high-road (1996/2000)
2. Pessoa & Adolphs 2010, "from a 'low road' to 'many roads'", Nat Rev Neurosci — https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2015.00101/full
3. McFadyen et al. 2019, pulvinar–amygdala white-matter pathway, eLife — https://elifesciences.org/articles/40766
4. Colliculus–pulvinar–amygdala encodes negative emotion — https://pmc.ncbi.nlm.nih.gov/articles/PMC8349850/
5. Rapid subcortical amygdala route for faces, J Neurosci 2017 — https://www.jneurosci.org/content/37/14/3864
6. Rapid processing of invisible fearful faces in human amygdala, J Neurosci 2023 — https://www.jneurosci.org/content/43/8/1405
7. McFadyen 2019, subcortical route across species (review) — https://journals.sagepub.com/doi/10.1177/1179069519846445

**分布式 / 非模块 / 建构**
8. Pessoa, *The Cognitive-Emotional Brain* — Dual Competition Model — https://mitpress.universitypressscholarship.com/view/10.7551/mitpress/9780262019569.001.0001/upso-9780262019569-chapter-7
9. Barrett 2017, theory of constructed emotion (SCAN) — https://pmc.ncbi.nlm.nih.gov/articles/PMC5390700/
10. Barrett et al. 2025, "Theory of Constructed Emotion: More Than a Feeling" — https://journals.sagepub.com/doi/10.1177/17456916251319045
11. Saarimäki et al., distributed affective space across the brain — https://www.biorxiv.org/content/10.1101/123521.full.pdf
12. Touroutoglou et al., large-scale affective/social networks — https://pmc.ncbi.nlm.nih.gov/articles/PMC4119963/
13. Kragel & LaBar 2013, autonomic patterns (via [9])

**评价 (CPM)**
14. Sander, Grandjean & Scherer 2018, appraisal-driven componential approach — https://journals.sagepub.com/doi/10.1177/1754073918765653

**内感受 / allostasis**
15. Barrett & Simmons 2015, "Interoceptive predictions in the brain" (EPIC), Nat Rev Neurosci — https://www.nature.com/articles/nrn3950
16. Kleckner et al. 2017, large-scale allostasis & interoception system — https://www.biorxiv.org/content/10.1101/098970v1.full

**整合：全局工作空间 / 显著网络 / 同步 / 价值**
17. Mashour, Roelfsema, Changeux & Dehaene 2020, Global Neuronal Workspace, Neuron — https://www.researchgate.net/publication/339708346_Conscious_Processing_and_the_Global_Neuronal_Workspace_Hypothesis
18. The global neuronal workspace as a broadcasting network (Network Neuroscience 2022) — https://direct.mit.edu/netn/article/6/4/1186/111960/The-global-neuronal-workspace-as-a-broadcasting
19. Predictive global neuronal workspace — active inference model — https://www.sciencedirect.com/science/article/abs/pii/S0301008220301738
20. Menon & Uddin 2010, "Saliency, switching, attention and control: insula network" — https://pmc.ncbi.nlm.nih.gov/articles/PMC2899886/
21. Triple network model — SN/DMN/FPN interactions — https://pmc.ncbi.nlm.nih.gov/articles/PMC5647507/
22. Fries 2015, "Rhythms for Cognition: Communication through Coherence" — https://pubmed.ncbi.nlm.nih.gov/26447583/
23. Schultz, dopamine reward prediction error coding (2016) — https://pubmed.ncbi.nlm.nih.gov/27069377/
24. Dopamine RPE hypothesis & cortico–basal-ganglia broadcast — https://pmc.ncbi.nlm.nih.gov/articles/PMC3176615/

**生存回路 / 情绪意识（快通路降级依据）**
25. LeDoux & Brown 2017, "A higher-order theory of emotional consciousness", PNAS — https://www.pnas.org/doi/10.1073/pnas.1619316114
26. The New LeDoux: survival circuits & the surplus meaning of 'fear' — https://academic.oup.com/pq/article-abstract/70/281/809/5802834
27. Are LeDoux's survival circuits basic emotions under another name? — https://www.sciencedirect.com/science/article/abs/pii/S2352154618300299

**GWT→AI 工程对照**
28. Theater of Mind: GWT 认知架构 for LLMs (2026) — https://arxiv.org/html/2604.08206v1
29. GWT 六个可测标记 (2026) — https://www.researchgate.net/publication/400184630_Evaluating_Global_Workspace_Markers_in_Contemporary_LLM_Systems
30. Free Energy in a Circumplex Model of Emotion (2024) — https://arxiv.org/html/2407.02474v1
