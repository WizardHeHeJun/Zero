# 给「文本输出」补足类人情绪——生物学 + 数学的文献调研与工程落点

> **日期**：2026-06-24
> **性质**：一次本地学术调研的成果固化（承接 `2026-06-23-emotion-math-and-llm-expression.md` 的总论，本篇收窄到**生成的语言/文本输出**这一具体通道）。
> **生成方式**：基于 WebSearch / WebFetch 检索的同行评审文献 + arXiv 预印本整理；公式为标准结果的逐步展开。
> **核对说明**：文献依检索结果给作者/年份/链接与核心结论；若用于正式论文/报告，建议逐篇核实出处与具体数据。
> **与本仓库的关系**：直接对应目标系统的语言层（`src/agents/language.py` 的 `LanguageAgent` + 双向收敛回路、`src/agents/affect_math.py` 的情感内核）。第五节给可执行落点。

---

## 目录

1. [问题定位：文本输出通道的现状与缺口](#一问题定位文本输出通道的现状与缺口)
2. [生物学侧：人脑怎么生成「带情绪的语言」](#二生物学侧人脑怎么生成带情绪的语言)
3. [数学/计算侧：把 e\*=(v,a) 落到 token 的四代方法](#三数学计算侧把-evae-落到-token-的四代方法)
4. [方法谱系总表（按「在哪注入情感」）](#四方法谱系总表按在哪注入情感)
5. [工程补足清单（落到本仓库）](#五工程补足清单落到本仓库)
6. [三道鸿沟（承接旧笔记，不重复展开）](#六三道鸿沟承接旧笔记)
7. [完整文献来源](#七完整文献来源)

---

## 一、问题定位：文本输出通道的现状与缺口

当前文本通道：情感内核采样出 `e* = (valence, arousal)` → `text_label` 映射到 **4 个离散词**（excited / content / angry / sad，`affect_math.py`）→ `_TemplateLanguageModel` 套模板 `[label] 关于 topic`（`language.py`）。即便接 `OpenAILanguageModel`，情感对文本的影响也只是 prompt 里的一个标签 + 二段式 VAD 反推。

对照文献，人类「带情绪的语言」有三样东西这里缺：

| 缺口 | 文献依据（见后） | 后果 |
| --- | --- | --- |
| **①内核驱动**：情感应改写遣词/句法/节奏，而非末端贴标签 | 情感语言产出脑回路；躯体标记 | 文本情绪是「挂上去」的，不是「长出来」的 |
| **②概念粒度**：4 档离散太扁，人类情绪词汇丰富 | 构建论 / 情绪粒度 | 低粒度 = 机器味 |
| **③个体史 / 调节**：基调、立场、重评 vs 抑制 | Gross 过程模型；心境动力学 | 缺时间深度与「克制/流露」之分 |

---

## 二、生物学侧：人脑怎么生成「带情绪的语言」

### A. 情感的「源头」是身体——躯体标记 + 内感受

Damasio 躯体标记假说：情绪本质是身体状态改变（心率、内脏、姿态…），经岛叶/体感皮层表征，在 vmPFC/杏仁核与过往结果绑定，**在词/决策被选出之前就偏置认知**（body-loop）。与旧笔记自由能的「内感受预测误差→情感」同源。
**对文本含义**：人类带情绪的文字背后有「身体证据」；项目里 `physiology_decoder`、arousal 已在算身体侧，但是单向 e\*→生理，应让生理/内感受反过来约束文本。

### B. 情感的「种类结构」——三大阵营，对文本是互补不是二选一

| 阵营 | 代表 | 核心主张 | 对文本输出的用途 |
| --- | --- | --- | --- |
| 离散/基本情绪 | Ekman | 若干普遍、先天、独立神经系统的基本情绪，可混合 | 给「情绪类别词」根基 |
| 皮层下情绪系统 | Panksepp | 七大进化保守环路：**SEEKING / RAGE / FEAR / CARE / PANIC-GRIEF / LUST / PLAY** | 在 VA 上加一层**动机色彩** |
| 维度 | Russell circumplex | 情感是 VA 连续空间的点 | **项目内核就用这个**——连续可控 |
| 构建论 | Barrett / Lindquist | 语言用概念给身体信号「赋义」，决定**情绪粒度** | 表层用词细分 |

离散 vs 维度之争至今未定论；「独立神经系统支持离散基本情绪」的证据被评为不一致。
**工程结论**：维度做内核（连续可控）＋ 离散/动机做表层粒度（人类用词），二者叠加——正补 4 档离散词的扁平。

### C. 情感的「语言产出回路」——边缘↔语言接口 + 神经调质映射

情感语言产出招募 IFG / 颞上回 / **岛叶 / 基底节 / ACC** 分布式网络；ACC 在情绪语义有歧义时做**冲突监控**；情感**改写词汇选择、句法、语速**。

**神经调质的计算映射**（与项目数学内核几乎一一对应）：

| 神经调质 | 计算角色 | 项目对应（`affect_math.py`） |
| --- | --- | --- |
| 多巴胺 DA | 奖励预测误差（TD） | `td_update` 的 `delta`/RPE |
| 5-羟色胺 5-HT | 奖励时间尺度 / 不确定性下学习率 | `gamma` / `lr`（可做自适应） |
| 去甲肾上腺素 NE | 行动随机性 / 增益 / 唤醒 | `precision`/π、arousal |

项目内核已经是这套的计算代理，只是没点明语义——补注释即可让架构自洽。

### D. 情感的「调节」——Gross 过程模型

**重评（reappraisal，早期：改变对情境的解释）**全面优于**抑制（suppression，晚期：压住外在表达）**；抑制只改表达不改体验、耗认知资源。
**对文本含义**：`appraisal.py` / `regulation.py` 恰好是这两个干预点——文本情绪强度该靠**改 OCC 评价**（重评）调，而非末端**砍输出**（抑制），前者更自然、更像人。

---

## 三、数学/计算侧：把 e\*=(v,a) 落到 token 的四代方法

### 路径 1 · 词典桥（最易接，无需开放权重）

用词级 VAD 规范库把候选词 \(w\) 映射成 \(\phi(w)=(v_w,a_w)\)——Warriner 13,915 词、NRC-VAD v2 **55k+ 词/短语**。做**加权解码**，把内核 e\* 当对 token 分布的偏置：

\[
\text{logit}'(w)=\text{logit}(w)+\beta\,\langle \phi(w),\,e^\*\rangle,\qquad P'(w)\propto P(w)\,e^{\beta\langle\phi(w),e^\*\rangle}
\]

\(\beta\) = 情感强度旋钮，可由 arousal 或精度 π 驱动。让「用词」本身被情感拉动，对应生物学侧 A。

### 路径 2 · 表示空间 steering（最机制化，开放权重）

LLM 激活空间里**情绪本身排成 VA 环**（circumplex），可线性操控：对比情绪文本与中性基线得情绪向量 \(v_e^{(\ell)}=\overline{h_e^{(\ell)}}-\overline{h_{\text{neutral}}^{(\ell)}}\)，PCA + 岭回归恢复 valence/arousal 轴 \(w_V,w_A\)（Llama-3.1-8B 第 31 层，valence 恢复 r=0.97、arousal r=0.87）。给定目标 (V,A)，按强度 α：

\[
\text{steering}^{(\ell)}=\alpha\,(V\!\cdot\! w_V+A\!\cdot\! w_A),\qquad h^{(\ell)}\leftarrow h^{(\ell)}+\text{steering}^{(\ell)}
\]

实测 valence 轴单调对称（VAD-BERT ±0.75）、arousal 轴近独立（valence 泄漏极小），直接改写首 token log-odds。历史源头 = sentiment neuron（Radford 2017，单神经元=情感 dial，覆写即控）。**对项目**：e\*=(v,a)∈ℝ² 即天然 steering 坐标，VA circumplex 内核与 LLM 内部 VA 几何同构。

### 路径 3 · 四代可控生成方法谱系

| 代 | 注入点 | 代表方法 | 机制要点 | 需开放权重 |
| --- | --- | --- | --- | --- |
| ① 训练期条件 LM | 模型参数 | **Affect-LM**（Ghosh 2017）；**ECM**（Zhou 2018） | Affect-LM 给 LM 加 β·affect 项，β∈[0,∞) 调强度（**高 β 牺牲语法**）；ECM 情绪类别嵌入 + **internal memory**（情绪值随句子 sigmoid 门衰减到 0）+ external 情绪词表 | 是 |
| ② 解码期可控 | 输出分布 | 加权解码（NRC-VAD）；**PPLM**（属性分类器梯度推 hidden）；**GeDi**（判别器引导）；**DExperts**（专家/反专家 product-of-experts）；**FUDGE** | 不改权重，外挂分类器/专家重排 logit | 否（API logit_bias 即可） |
| ③ 表示空间 steering | 隐状态 | sentiment neuron；style/emotion vectors；**VA 子空间环状几何** | \(h\leftarrow h+\alpha(Vw_V+Aw_A)\) | 是 |
| ④ 偏好/评价对齐 | 目标函数/前件 | RLHF/DPO 情感对齐；**APTNESS/EmPO**（共情）；**CPM**（Scherer）/**EMA**（Marsella & Gratch）评价驱动 NLG | 把「情商/评价过程」显式建模为前件/奖励 | 看实现 |

**两个值得借的细节**：
- **ECM internal memory 衰减**：句首「满情绪」、经 sigmoid 门递减至句尾归零 = 一句话内情绪释放的时间包络 → 可接 mood/arousal。
- **Affect-LM 的 β 折中**：强度越大、语法越烂 → steering/加权强度要设**语法保真上限**。

### 评测闭环（项目现缺）

要敢说「更像人」就得有度量：**EmoBench**（ACL24，理解+应用，中英 400 题）、**EQ-Bench**、**EmotionBench**、**EmoBench-M**（多模态）。

---

## 四、方法谱系总表（按「在哪注入情感」）

```
情感内核 e*=(v,a)
   │
   ├─ 训练期 →  条件 LM（Affect-LM β / ECM 类别嵌入+内存衰减）          [改权重]
   ├─ 解码期 →  加权解码（NRC-VAD 词典桥）/ PPLM / GeDi / DExperts       [API 可]
   ├─ 表示空间→ steering vector  h += α(V·w_V + A·w_A)（VA 环状几何）     [开放权重]
   └─ 偏好对齐→ RLHF/DPO + 评价驱动 NLG（CPM/EMA/APTNESS）+ EmoBench 评测  [看实现]
```

五块拼成「把情绪补到文本输出」的完整地图：维度（VA）做连续可控内核、离散+动机（Panksepp/构建论）做有粒度的表层用词、四代方法提供注入手段、神经调质/评价/调节理论提供参数语义与干预点、情商评测提供度量闭环。

---

## 五、工程补足清单（落到本仓库）

| 优先级 | 补足项 | 落点 | 依据 |
| --- | --- | --- | --- |
| ★高 | 粒度升级：`text_label` 4 档 → **VA 扇区 × Panksepp 动机色彩**二维词表（新增，不动 `text_label` 保零回归） | `src/agents/emotion_lexicon.py`（新）→ `language.py` | 构建论 + Panksepp |
| ★高 | NRC-VAD 加权解码桥：词级 VAD + `affect_logit_bias` 偏置/重排，强度由 π/arousal 驱动、设语法保真上限 | `emotion_lexicon.py` + `language_openai.py` | NRC-VAD + Affect-LM β |
| ★中 | **情绪时间包络**：一句话内强度按 sigmoid 衰减 | `emotion_lexicon.py` → `mood.py`/`language.py` | ECM internal memory |
| ★中 | VA steering 适配器（开放权重），e\* 直接当 (V,A) | 新增 `language_steering.py` | 2604.03147 / sentiment neuron |
| ★中 | 调节走**重评优先**（改 OCC 评价），不在末端砍输出 | `regulation.py`/`appraisal.py` | Gross |
| ★低 | π/TD/arousal 显式标注为 DA/5-HT/NE 计算类比（补语义注释） | `affect_math.py` | 神经调质映射 |
| ★低 | 评价条件化：传 OCC 评价向量而非只传 (v,a) | `appraisal.py`→`language.py` | CPM / EMA / APTNESS |
| ★低 | 加情商评测回归（EmoBench/EQ-Bench 子集） | `tests/` | LLM EI 评测 |

**本次执行落地**：新增纯函数模块 `src/agents/emotion_lexicon.py`（VA 扇区细粒度词表 + Panksepp 动机色彩 + 种子 VAD 词典 + 加权解码偏置 + ECM 式情绪时间包络），接入 `_TemplateLanguageModel`（丰富默认文本）与 `OpenAILanguageModel`（可选词汇提示，默认关），补 `tests/test_emotion_lexicon.py`。全部 torch-free / API-free / 默认关 / 零回归。

---

## 六、三道鸿沟（承接旧笔记）

扩大检索未改变三道鸿沟：**效度**（VAD/离散标注有争议、跨标注者一致性低）、**坍缩**（压成 (v,a)/标签丢信息）、**qualia**（steering 让文本「读着像 X 情绪」≠「有 X 情绪」）。以上全部是补**情感的功能投影在文本上的表现**，不是「让系统真的有感受」。

---

## 七、完整文献来源

**生物学侧**
- 躯体标记 / 内感受：[Somatic Marker（Wikipedia）](https://en.wikipedia.org/wiki/Somatic_marker_hypothesis) · [body-loop 综述](https://www.sciencedirect.com/science/article/abs/pii/S2352154617300736)
- 情感语言产出脑回路：[情感韵律产出 fMRI（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC5067951/) · [情感与感觉运动成分（J. Neurosci.）](https://www.jneurosci.org/content/33/4/1640)
- 情绪种类结构：[Panksepp 七系统（Frontiers）](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.01025/full) · [Ekman 基本情绪](https://sk.sagepub.com/ency/edvol/download/the-sage-encyclopedia-of-theory-in-psychology/chpt/ekman-s-theory-basic-emotions.pdf) · [离散 vs 维度对比（Frontiers）](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1287334/full)
- 构建论 / 情绪粒度：[Lindquist & Barrett: 语言在情绪中的角色（Frontiers）](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00444/full) · [构建与记忆的预测编码视角](https://www.researchgate.net/publication/340036299_The_Role_of_Language_in_the_Construction_of_Emotion_and_Memory_A_Predictive_Coding_View) · [文本情绪粒度作为心理健康指标](https://arxiv.org/pdf/2403.02281) · [构建情绪理论（Wikipedia）](https://en.wikipedia.org/wiki/Theory_of_constructed_emotion)
- 神经调质计算映射：[Doya 神经调质元学习](https://www.sciencedirect.com/science/article/abs/pii/S0893608002000448) · [NE 威胁预测误差](https://pmc.ncbi.nlm.nih.gov/articles/PMC11269024/) · [神经调质系统综述（Frontiers）](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2017.00108/full)
- 情绪调节：[Gross 过程模型（PubMed）](https://pubmed.ncbi.nlm.nih.gov/12212647/)

**数学/计算侧**
- 词典桥：[Warriner 2013（13,915 词）](https://www.researchgate.net/publication/235604845_Norms_of_valence_arousal_and_dominance_for_13915_English_lemmas) · [NRC-VAD v2（55k+ 词）](https://arxiv.org/abs/2503.23547)
- 表示空间 / steering：[VA 子空间环状几何（arXiv 2604.03147）](https://arxiv.org/html/2604.03147v3) · [Anthropic: LLM 中的情绪概念](https://transformer-circuits.pub/2026/emotions/index.html) · [Style Vectors](https://arxiv.org/html/2402.01618v1) · [sentiment neuron（Radford 2017）](https://arxiv.org/abs/1704.01444) · [小模型情绪表示抽取与 steering](https://arxiv.org/html/2604.04064)
- 可控生成四代：[Affect-LM（ACL 2017）](https://aclanthology.org/P17-1059/) · [ECM（AAAI 2018）](https://arxiv.org/abs/1704.01074) · [PPLM](https://arxiv.org/abs/1912.02164) · [DExperts](https://www.semanticscholar.org/paper/DExperts:-Decoding-Time-Controlled-Text-Generation-Liu-Sap/02f033482b8045c687316ef81ba7aaae9f0a2e1c) · [可控生成综述](https://arxiv.org/pdf/2201.05337)
- 评价驱动 NLG / 共情：[CPM（Scherer）](https://philpapers.org/rec/SCHTCP-10) · [EMA（Marsella & Gratch）](https://www.sciencedirect.com/science/article/abs/pii/S1389041708000314) · [APTNESS](https://arxiv.org/html/2407.21048v1) · [EmpatheticDialogues（ACL）](https://aclanthology.org/2022.findings-emnlp.340/) · [EmPO](https://arxiv.org/pdf/2406.19071) · [计算情感建模综述（CACM）](https://dl.acm.org/doi/10.1145/2631912)
- LLM 情商评测：[EmoBench（ACL24）](https://arxiv.org/abs/2402.12071) · [EQ-Bench](https://arxiv.org/pdf/2312.06281) · [EmoBench-M](https://arxiv.org/abs/2502.04424)
