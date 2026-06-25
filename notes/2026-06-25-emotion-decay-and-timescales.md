# 情绪的衰退机制 + 情绪/态度的时间尺度分层

> **日期**：2026-06-25
> **性质**：一次情感科学文献调研的成果固化（承接对话耦合 chat 的真机迭代）。
> **缘起**：真机对话里发现 chat 把情绪当成**单一慢积分器**——"恼火"在用户转缓后仍赖着不走。用户指出："一种情绪不应该是长期积累的结果，除非是一种长期印象会积累对某一件事某一个人的态度。" 本篇查文献证实并落成**两时间尺度**模型。
> **生成方式**：WebSearch 同行评审文献整理；核对说明同前几篇。
> **落点**：`affect_math.emotion_decay_step` / `attitude_step` + `main.py --chat`。

---

## 一、问题：情绪不是长期累积量

旧实现 `feeling = leaky_integrator(e*)`（inertia 0.7）把"情绪"建成一个慢积分器：被骂累积变负是对了，但**停止被骂后衰退太慢**，怒气数轮不散。这违反情感科学的基本事实——情绪是**短时**的。

---

## 二、情绪是短时的，会衰退回基线（affective chronometry）

**Davidson 的 affective chronometry**：情绪反应有三参数——**rise-time**（起始到峰值）、**amplitude**（峰值幅度）、**recovery/decay time**（回到基线的时长）[1][2]。核心断言：**"所有情绪状态都会自然衰退、回归机体的基线稳态"**[2]。**recovery time（回基线速度）是情感风格的关键标记**[3]。

**emotional inertia（情绪惯性，Kuppens）**：情绪状态过度延续、不易重置到中性，是**心理病理标志**（抑郁等）[2][4]。即——**衰退太慢本身不健康**。旧实现的"赖着不走"恰是病理化的惯性。

→ 工程含义：情绪应有**快衰退**（短 recovery time），刺激停止后几轮回基线。

---

## 三、三层时间尺度：情绪 < 心境 < 态度（Scherer / Frijda）

Frijda & Scherer 的"设计特征"区分情绪与其它情感状态[5][6]：

| 层 | 时长 | 触发 | 指向 | 例 |
| --- | --- | --- | --- | --- |
| **情绪 emotion** | 秒~分（短） | 特定事件 | 对象指向、有行动倾向 | 一阵怒 |
| **心境 mood** | 时~天 | 无特定事件 | 弥散、无指向 | 一整天闷闷的 |
| **态度/sentiment** | 最久、稳定 | 非事件触发 | **对特定对象**的稳定评价 | 讨厌某人 |

"情绪短时、强烈、事件触发；心境弥散、更久；**sentiment 不那么急、持续最久、对象指向**"[5]。**态度 = 对某对象的稳定 favorable/unfavorable 评价**[6]。

→ 正是用户说的：长期累积的不是情绪，是**对某人/某事的态度**。

---

## 四、计算模型现成：ALMA / WASABI（情绪衰退 + 累积成心境/态度）

- **ALMA（Gebhard 2005，A Layered Model of Affect）**[7]：三层 **Emotion(短，OCC) → Mood(中，PAD，= "情绪状态的平均") → Personality(稳定)**。**"情绪强度有自然衰退（可配置衰退函数）"**；Mood 是情绪的移动平均、被活跃情绪推拉（Mehrabian "mood = 情绪状态在各情境的平均"）。
- **WASABI（Becker-Asano）**[8]：**"所有效价自行稳步衰减直到回到中性"**（情绪衰退）；**"任何正/负情绪效价随时间正/负向影响心境"**（情绪→心境累积）。"情绪=瞬时；心境=持久、情绪的累积效应；人格=稳定"。

→ 现成范式：**快变情绪自然衰退到基线 + 慢变量累积自情绪流**。

---

## 五、态度怎么来：evaluative conditioning（重复情感经验 → 对象态度）

**Evaluative conditioning（EC）**：中性刺激经与带效价刺激**重复配对**，获得正/负评价反应 → **对该对象形成/改变态度**[9][10]。"对某对象的**重复情感经验累积**影响总体 sentiment""重复配对使对象被感到正/负"。Fazio：**态度 = 对象-评价关联，强度随经验积累**[11]。

→ 对话里：对"这个人"的态度 = 与 ta 互动的**情绪流的慢累积**（每轮 e* 喂一点）。对象 = 对话方（按会话线程绑定）。

---

## 六、落地模型（两时间尺度，`affect_math` 纯函数）

```
# 慢：态度（对此人的长期印象），evaluative conditioning，多轮才成形、持久化
attitude' = (1-rate)·attitude + rate·e*          rate=0.08（≈10+轮成形）

# 快：情绪（短时），affective chronometry——向 attitude 基线衰退 + 当前刺激冲击 + 噪声
emotion'  = attitude + recovery·(emotion-attitude) + reactivity·e* + noise
           recovery=0.4（残留比例小=回基线快，~2-3轮）  reactivity=0.6
```

- **情绪短时**：刺激停（e*≈0）→ deviation 每轮 ×0.4 衰减、几轮回到 `attitude` 基线（不长期累积；过慢=惯性病理）。
- **态度慢、对人**：单句几乎不动；持续负面多轮才把基线压冷；**只持久化 attitude**（情绪重启即归基线）。
- **基线 = 态度**：怒火退去后回到"对此人的态度"而非绝对中性——持续被骂的人，基线变冷、再聊也带刺；偶尔被呛的人，怒一下就过、态度不塌。
- 表达取 `emotion`（当前短时情绪）→ `affect_label` 多样词 → `converse`。
- **真机验证**（qwen-flash）：辱骂→怒火飙(警觉)→道歉即回正(欣喜/兴奋)；两句辱骂没让态度永久变冷（需持续）。

---

## 七、三道鸿沟（承接旧笔记，未跨越）
衰退/时间尺度都是对情绪**功能时程**的更忠实建模；效度（VAD 标注争议）、坍缩、qualia 依旧未解。

---

## 八、文献来源

**affective chronometry / 情绪衰退**
1. Still feeling it: time course of emotional recovery (Frontiers Hum Neurosci 2013) — https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00201/full
2. Temporal dynamics of affect in the brain (PMC8792368) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8792368/
3. Temporal dynamics of emotional responding: amygdala recovery predicts emotional traits (SCAN) — https://academic.oup.com/scan/article/9/2/176/1623479
4. Temporal dynamics of spontaneous emotional brain states (PMC9026845) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9026845/

**情绪/心境/态度时间尺度（Scherer/Frijda）**
5. Distinctions between Emotion and Mood (ResearchGate) — https://www.researchgate.net/publication/32116456_Distinctions_between_Emotion_and_Mood
6. Differentiating Emotions from Other Constructs (Psych of Human Emotion, OA textbook) — https://psu.pb.unizin.org/psych425/chapter/differentiating-emotions-from-other-constructs/

**计算模型（衰退 + 分层）**
7. Gebhard, ALMA – A Layered Model of Affect (AAMAS 2005) — https://alma.dfki.de/papers/aamas05.pdf
8. Becker-Asano, WASABI: Affect Simulation for HCI — https://www.becker-asano.de/becker-asano_ERM4HCI.pdf

**态度形成（evaluative conditioning）**
9. Evaluative conditioning (Wikipedia) — https://en.wikipedia.org/wiki/Evaluative_conditioning
10. The Role of Evaluative Conditioning in Attitude Formation — https://www.researchgate.net/publication/254081472_The_Role_of_Evaluative_Conditioning_in_Attitude_Formation
11. Attitudes as Object-Evaluation Associations of Varying Strength (Fazio, PMC2677817) — https://pmc.ncbi.nlm.nih.gov/articles/PMC2677817/
