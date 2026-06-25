# 双路语言：皮层 LLM 出内容 + 皮层下情绪 push 漏出来（不替换、只辅佐）

> **日期**：2026-06-25
> **性质**：神经科学文献调研 + 架构定调（承接对话耦合）。
> **缘起**：用户洞察——"情感网络是**辅佐**内容产生、不替换 LLM；神经科学里语言由皮层直接控制，而很多情绪是皮层下其他脑区**自动**产生的不随意行为；LLM≈皮层管理性语言，情感系统≈皮层下管感性着色。"本篇查文献证实并定为架构原则。
> **生成方式**：WebSearch 同行评审文献；核对说明同前。

---

## 一、结论：情绪表达走「皮层下·自动·不随意」通路，与「皮层·随意·命题」语言是两套独立系统

这是**双通路**（dual-route）的硬结论，在**发声**和**面部**上都成立：

### A. 发声双通路（Jürgens）[1]
- **皮层·随意**：喉运动皮层 → 直接皮质运动神经元 → **命题性言语**（理性"说什么"）。
- **边缘·不随意**：前扣带回(ACC)+杏仁核+下丘脑 → 中脑导水管周围灰质(PAG) → **情绪性发声**（自动）。
- 经**额斜束**(ACC↔喉皮层)整合。→ 语言皮层、情绪声音皮层下，分开。

### B. 面部双通路（锥体 vs 锥体外系）[2]
- **锥体·随意**：初级运动皮层 → 摆拍假笑。
- **锥体外·自发**：皮层下(基底节/脑干网状) → 面神经核 → 真情绪 Duchenne 笑。
- **双重分离**（临床）：岛叶梗死→自发在、随意没了；反向→情绪没了、随意在。→ 两套真独立。

### C. 情绪可绕过认知自动产生 [3]
Zajonc 情感优先："几乎无需认知加工"即唤起情感；LeDoux 低路"在皮层处理前"触发。（现主流为双向互动，非纯优先。）

### D. Scherer push vs pull [4]
- **push**：情绪的**不随意生理改变**直接改变表达（情绪"泄漏"）。
- **pull**：社会显示规则**随意**地把表达拉向目标。
→ 情感通道=push(不随意)；理性/社会=pull(随意)；输出是两者叠加。

---

## 二、对应到本系统（LLM=皮层/pull，情感引擎+自动通道=皮层下/push）

| 神经科学 | 本系统 |
| --- | --- |
| 皮层·随意·命题·pull·"说什么" | **LLM**（理性内容） |
| 皮层下·自动·情绪·push·"怎么泄漏" | **情感引擎 + steering / 韵律 / 自发表情** |
| 额斜束整合 | (v,a) 在输出端融合两路 |

**"辅佐不替换"成立**：情感不接管语言，而像皮层下通路那样**自动调制/泄漏**到输出。

---

## 三、关键洞察：这解释了之前的「扮演」问题

让 LLM "你愤怒，请表达" → LLM **有意识/皮层地编排**愤怒 = 把情绪塞进**随意/锥体/pull** 通路 → "摆拍/表演"（早先用户批评的）。

**忠实做法 = 走皮层下/push**：情绪不经"我决定要演"，而**自动调制**皮层产物：
- **解码期 push（API 可，零开放权重）**：`emotion_lexicon.affect_logit_bias` 给 affect-congruent 词加 `Δlogit=β·⟨φ(w),e*⟩` → 用词分布在解码层被情绪偏置，**不靠"演"的指令**。
- **表示空间 push（开放权重）**：`SteeringLanguageModel` 把 e\* 向量加到隐状态——最像锥体外系。
- **不随意通道**：`ProsodyDecoder`/`expression.py` 自发头直接解码 e\*，**绕过 LLM**。

`expression.py` 早有面部双通路（spontaneous 皮层下 / voluntary 皮层）——本架构把**同一原则推广到语言**。

---

## 四、落地（`--chat` 接 push）

1. **解码期 push（默认，零依赖）**：converse 把 `suggest_affect_words(e*)` 作**不随意用词倾向**注入（"状态自然流露、别点破别表演"），与"理性内容"分离——情绪进**用词/节奏**而非"演情绪"；可选叠加 OpenAI `logit_bias`（需兼容 tokenizer/模型，graceful 回退）。
2. **表示空间 push（可选）**：本地开放权重 LM + `SteeringLanguageModel`。
3. content 仍由 LLM 出（皮层/pull），push 只着色（皮层下）。

---

## 五、诚实的边界
不是非黑即白：皮层也做**自上而下情绪调节**（评价是皮层的；pull 显示规则是皮层/社会的）；Pessoa "many roads" 提醒通路不止两条。故准确说是**双路 + 整合 + 皮层自上而下调节**。API 黑盒 LLM 上"纯 push"难做到极致（真 push 需 logit_bias/开放权重）；prompt 级词倾向是近似（偏向 form 而非 content）。三鸿沟未跨越。

---

## 六、文献来源
1. Jürgens, Neural pathways underlying vocal control (Neurosci Biobehav Rev 2002) — https://www.researchgate.net/publication/11504415_Jurgens_U_Neural_pathways_underlying_vocal_control_Neurosci_Biobehav_Rev_26_235-258 · Two motor systems for human speech (J Comp Neurol) — https://onlinelibrary.wiley.com/doi/abs/10.1002/cne.23898
2. Facial Expressions: Pyramidal & Extrapyramidal (Purves Neuroscience/NCBI) — https://www.ncbi.nlm.nih.gov/books/NBK10829/box/A2032/ · Amygdalo-motor pathways & facial expression (PMC3958699) — https://pmc.ncbi.nlm.nih.gov/articles/PMC3958699/
3. Zajonc, primacy of affect — https://deepblue.lib.umich.edu/handle/2027.42/104016 · On the interdependence of cognition and emotion (PMC2366118) — https://pmc.ncbi.nlm.nih.gov/articles/PMC2366118/
4. Scherer, Vocal Expression of Emotion (push/pull) — https://www.diva-portal.org/smash/get/diva2:165425/fulltext01.pdf · Symbolic Functions of Vocal Affect Expression (Scherer 1988) — https://journals.sagepub.com/doi/abs/10.1177/0261927X8800700201
