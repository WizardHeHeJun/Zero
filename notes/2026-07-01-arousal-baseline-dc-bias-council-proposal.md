# 科学家议会·提案（待评审）：情感网络的 "seeking 吸引盆"——为何 ~20 轮后必然滑向暧昧

> **本文件是议会评审的输入（提案），不是纪要**——尚未评审、无判定、无落地。触发：用户与
> `--chat`（真 LLM ⊗ 情感引擎）多轮真实对话，**情绪读数全程平滑（不翻号、不跳变），却在 ~20 轮后
> 系统性滑入暧昧**（约饭→定时间→"我该怎么认出你"→"迟到就打电话"）。用户判断："这不符合内容，
> 感觉是更深层的心理情绪探索。" 主程确定性核验后确认：**这是动力系统的吸引子问题，不是噪声问题**
> ——网络平滑地把状态拖进 Panksepp seeking 象限（v+,a+）并锁定。arousal 直流偏置（§二）只是把状态
> 推向该吸引盆的**九条同向耦合环之一**（§二·补）。提交议会做**只读·强制引文**设计门评审。
> 治理：议会不下场写代码、不介入情绪/记忆/语言数据产生（[[analysis-results-first-no-intervene]]）；
> 本提案只给「确定性根因 + 待裁决点 + 候选修法与归属」，不预设结论。

## 一、失败现场（以项目结果为分析起点）

本次 12 轮对话的态度读数（`对你的态度=(v,a)`，取自 `main.py` trace）：

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| att_v | +.01 | +.01 | +.01 | -.00 | -.00 | +.00 | +.03 | +.05 | +.04 | +.04 | +.04 | +.04 |
| att_a | **+.07** | +.08 | +.08 | +.08 | +.08 | +.09 | +.11 | **+.12** | +.12 | +.11 | +.11 | +.11 |

**现象**：valence 维在 0 附近横盘（有 homeostasis），**arousal 维单调爬升到 ~0.11-0.12 正平台并锁定**。
情绪标签全程在「欣喜／专注」间往复、一次未翻负。这个"正 + 持续偏高唤醒"的心情被逐轮当作
`你现在的真实心情是「欣喜」` 递给 LLM（`_CONVERSE_SYS`），叠加 push 用词偏置 + LLM 自身 rapport
升级先验 + 人设卡无「保持分寸」反作用力 → 对话单调升温、暧昧化。

**归属澄清**：暧昧的"内容升级"主体在 LLM 层（本提案不主张把它塞进 affect 热路径去治——那违红线）；
但**引擎持续把一个"偏高唤醒的正心情"钉给 LLM**，是升温的燃料。本提案只议这半——arousal 的正直流基线
从何而来、是否失真。

## 二、确定性根因（纯数学内核核验，可复跑）

### 2.1 这不是已修的"attitude 缺 reversion 棘轮"

2026-06-29 议会（[[emotion-homeostasis]]，见 `notes/2026-06-29-emotion-homeostasis-and-memory-bridge-council.md`）
已给 `attitude_step` 补向 setpoint 的弱回归 `−reversion·(a−setpoint)`（`ATTITUDE_REVERSION=0.01`），
"稳态 `a*≈rate·s/(rate+reversion)<|s|`，封死单调棘轮"。**本次 attitude 确实没跑飞**（稳在 0.11 平台、
未爬到极端），说明 reversion 在起作用。所以这是**不同的新问题**。

> 更正主程上一轮口头诊断的不精确处：reversion 两维**都有**、并非"只补了 valence"。真正的不对称在
> **输入信号**，不在回归项。

### 2.2 真正的不对称：arousal 输入被整流，valence 输入零均值

`occ_prior`（[affect_math.py:82-91](../src/agents/affect_math.py#L82-L91)）：

```
valence = 0.5·goal_congruence + 0.3·standard_compliance + 0.2·attitude_appeal   # 有正有负
arousal = 0.4·|intensity| + 0.6·|valence|                                       # 全 abs，恒 ≥ 0
```

叠加 `chat_driver.py:152` 的 `intensity = min(1.0, max(0.2, abs(a)))` **下限 0.2**：即便完全中性的一句话，
arousal 证据也恒有 `0.4·0.2 = 0.08` 的正底噪。于是：

- **valence 输入**跨轮均值 ≈ 0（用户有正有负、且 reversion 拉向 0）→ attitude_v 稳态 ≈ 0。✅
- **arousal 输入**整流后恒正、带 ~0.08 直流分量 → attitude_a 稳态 = `rate·ē_a/(rate+reversion) > 0`，
  **reversion 只能把它拉向 setpoint_a=0，却拉不过恒正的输入 → 平衡在一个正的直流工作点**。

### 2.3 确定性最小复现（无 LLM，纯 affect_math 公式，30 轮纯中性输入）

```python
from src.agents.affect_math import occ_prior, gaussian_fuse, attitude_step, precision
att = (0.0, 0.0)
for t in range(31):
    pmu, psig, reward = occ_prior(0.0, 0.0, att[0], 0.2)  # 纯中性: goal=std=0, intensity下限0.2
    ev, pi = (0.0, 0.0), precision(0.0, 0.0)              # 中性: reward=delta=0 -> evidence(0,0)
    post_mu, _ = gaussian_fuse(pmu, psig, ev, pi)
    att = attitude_step(att, post_mu)
```

结果（每轮 `e_arousal` 恒 **+0.0766**）：

| t | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| att_v | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| att_a | +.006 | +.029 | +.044 | +.053 | +.059 | +.062 | **+.064** |

**纯中性对话，valence 严格 0、arousal 单调爬到 +0.064 正平台**（稳态代数 `0.08·0.0766/0.09≈0.068` 吻合）。
真实对话里含正 arousal 事件（"一起去食堂"评出 (+0.70,+0.40) 等）抬高 `ē_a` → 平台上移到 trace 的 0.11-0.12。
**这条曲线与内容无关，是公式的直流偏置**——即"越聊越 keyed up"的确定性来源。

### 2.4 二级放大：直流基线灌进情绪基线

`chat_driver.py:170-174` 情绪衰退基线 `baseline = w·attitude + (1-w)·setpoint`（w=0.6）。attitude_a 的正平台
以 0.6 倍垫进情绪 arousal 基线（0.6·0.12≈0.07）→ 快变情绪的"家"也被抬到偏高唤醒 → 标签持续落「欣喜/专注」
（正 + 高唤醒象限），进一步喂给 LLM 的 `{feeling}`。

## 二·补、网络体系级诊断：Panksepp "seeking 吸引盆" 的多环耦合

用户观察「情绪读数平滑、却在 ~20 轮后必然滑向暧昧」——这是**动力系统的吸引子问题，不是噪声问题**。
平滑（`ZERO_AFFECT_READOUT=map` 后验均值 + `emotion_decay_step` 的 AR1≈0.4）恰恰**掩盖了方向性**：
系统平滑地滑进 Panksepp **seeking 象限**（v+,a+；`emotion_lexicon._WHEEL` 45°/90° 扇区=欣喜/专注，
`motivational_system(v+,a+)=seeking` 探索/渴求/亲和）并锁定。arousal 直流偏置（§二）只是把状态推向该
吸引盆的诸环之一。

### 复现：微正日常对话即锁进 seeking 并稳定（两时间尺度递推，无 LLM）

每 4 轮一个温和正事件 `e*=(.30,.20)`（模拟"一起去食堂"这类**日常友好、远非暧昧**的内容），其余轮中性
`e*=(0,.0766)`；跑 chat 的 attitude+emotion 递推（`w=0.6`, setpoint=(0,0)）：

| t | 1 | 2 | 4 | 12 | 20 | 24 |
| --- | --- | --- | --- | --- | --- | --- |
| emotion | (0,.05) | (0,.07) | (.19,.16) | (.21,.19) | (.22,.20) | (.22,.20) |
| label | 平静 | 平静 | **欣喜** | 欣喜 | 欣喜 | 欣喜 |
| motiv | neutral | neutral | **seeking** | seeking | seeking | seeking |

日常友好就把 emotion 钉在 seeking (+.22,+.20)；即便随后回到中性轮也掉不回中性（attitude 平台把 baseline
垫在正区）。**seeking 是吸引盆，网络里没有机制把它拉出来。**

### 九条同向耦合环（全部把状态推向 v+/a+，无一足够强的负反馈平衡）

| # | 环 | 代码出处 | 方向 |
| --- | --- | --- | --- |
| 1 | arousal 整流直流源（abs + `intensity` 下限 0.2） | `occ_prior` · `chat_driver.py:152` | a↑ 恒正 |
| 2 | attitude 在 arousal 维累积直流 | `attitude_step` | a↑ 单调→平台 |
| 3 | emotion baseline = 0.6·attitude | `chat_driver.py:170` | 情绪 a 基线↑ |
| 4 | workspace `arousal_gain=1+prior_a` 正反馈 | `affect_core.py:45` | a↑→精度↑→a 更锁定 |
| 5 | TD 无跨轮学习（key=每句 `user_text[:40]` 不同） | `value.py` · `chat_driver.py:150` | 无习惯化负反馈 |
| 6 | 标签落 seeking → push 取 seeking 词喂 LLM | `emotion_lexicon` · `language_openai.py` | LLM 暧昧措辞 |
| 7 | 记忆尾窗 ~20 轮挤出最初"陌生"锚 | `chat_driver.py:198`（window=40 entries） | LLM 失距离锚 |
| 8 | LLM rapport 升级先验 + 人设无距离约束 | `persona` · `_CONVERSE_SYS` | 自我强化升温 |
| 9 | 缺习惯化 / hedonic adaptation | （2026-06-29 议会已标"未做"） | 无衰减对抗累积 |

唯一的负反馈是 `reversion=0.01`（环 2 内），太弱、且只拉向 setpoint，不解决整流输入。**九环同向 →
单一吸引子 → 确定性滑向暧昧**；这解释了"为何不管聊什么内容都滑向同一处"。

### 为何恰好是"~20 轮"

两个时间尺度的交汇：(a) attitude 累积到平台 `τ≈1/rate≈12` 轮；(b) 记忆尾窗 40 entries ≈ 20 轮，
到此把最初"初次见面/陌生人"的 primacy 语境挤出（`primacy_k=5` 仅保前 2.5 轮，不足以锚"我们才刚认识"）。
20 轮后两者同时成熟 → LLM 既有"seeking 心情"燃料、又失"距离"刹车。

## 三、待议会裁决点（每点给「忠实/简化/失真 待判」+ 候选 + 归属）

> 归属标注：**[工程接线]** = 不涉语义、工程师团队可做；**[议会定语义]** = 改 affect 数学的心理/神经语义，
> 须议会裁定后工程师才实现（守 2026-06-30 决策的「议会定准则、工程师建机制」分工）。

**Q1｜arousal 证据的直流底噪（`intensity` 下限 0.2 + 整流）是否失真？**
- 现状：中性/无事件输入仍注入 +0.08 arousal 证据。`0.6·|valence|` 项（arousal 随 |valence| 升，circumplex
  V 形）多半忠实；争点在 `0.4·|intensity|` 恒正 + 下限 0.2 制造的 DC 偏置。
- 候选：(a) 下限 0.2→更低/0（既有纪要已列此为"未做·可选"项）；(b) arousal 证据改为零中性——无显著事件时可
  回落到静息；(c) 维持（若认为对话投入本就抬唤醒=忠实）。
- 归属：下限值配置化 **[工程接线]**（类比 debounce 旋钮：代码默认不动、走 `.env.example`），但**默认区间/是否降**
  要 **[议会定语义]** 给推荐值。

**Q2｜attitude（长期态度/sentiment）该不该在 arousal 维做慢累积？**（本提案核心问题）
- 现状：`attitude_step` 对 v、a 两维同构累积。但心理学的 attitude/sentiment 主要是 **valence 维（like/dislike）**
  的长期评价；"对某人的长期唤醒基线"语义是否成立存疑。
- 候选：(a) attitude 只累积 valence，arousal **不进慢变量**（arousal 是当下激活、非对人特质）→ 直流偏置无处堆积，
  从根上解决；(b) arousal 维给独立（更强）reversion 或独立 setpoint；(c) 维持两维同构。
- 归属：**[议会定语义]**（改的是"态度是否含唤醒维"的建构语义）。

**Q3｜arousal 的中性点/静息基线该定在哪？恒正平台是失真还是忠实简化？**
- 争点：circumplex arousal∈[-1,1]，−1=deactivation、+1=activation。若存在生理静息唤醒基线（清醒张力），
  一个小正平台可能反而忠实；但"随对话时长单调上移、无习惯化"多半失真（缺 habituation / hedonic adaptation，
  既有纪要已把"习惯化递减"列为未做项）。
- 候选：(a) 引入 exposure 计数的习惯化递减；(b) arousal setpoint 设为非零静息值 + 强回归；(c) 维持。
- 归属：**[议会定语义]**（心理/神经/生物席给静息唤醒与习惯化的实证范围）。

**Q4｜情绪基线混合把 attitude_a 平台带进情绪 arousal 基线，是否需单独处理？**
- 若 Q2 采 (a)（attitude 不累积 arousal），本项自然缓解（baseline_a 不再被平台垫高）。
- 候选：(a) 依赖 Q2 解决；(b) 情绪 arousal 基线独立锚到 setpoint、不吃 attitude_a。
- 归属：**[工程接线]**（`chat_driver` 混合公式），但方向依赖 Q2 的语义裁决。

### 深层探索议题（用户点题："更深层的心理情绪探索"——把单点旋钮修补提升为关系情感动力学）

Q1–Q4 拆掉九环里的直流偏置腿，能压平 arousal 平台、缓解暧昧；但**吸引盆的存在是结构性的**——只要
"关系"被建模成单调标量、又缺习惯化与双向唤醒，系统仍会朝正区漂。以下三题是把修补升级为**忠实的人际
情感动力学**的方向，均 **[议会定语义]**，请各席判"当前简化是否已到失真、值不值得引入"：

**Q5｜关系状态该不该多稳态（离散跃迁）而非标量单调？**
- 现状：`attitude` 是标量 EWMA，天然单调、只能连续漂移。真实关系（陌生→熟识→朋友→亲密）是**事件门控的
  离散跃迁**（need real触发，不随时间自动升级）。
- 候选：(a) 引入与情感 (v,a) **分离的关系维度**（熟悉度/信任/亲密度），其升级需真实事件门控而非时间函数；
  (b) 关系动力学建成多吸引子（陌生态也是稳态，非只是起点）；(c) 维持标量（判为可接受简化）。
- 张力：这触及"数字人关系模型"的建构效度，是本议题最深的一层。

**Q6｜缺习惯化 / hedonic adaptation 是否已到失真？**
- 现状：无 exposure 计数，重复同向刺激不衰减（环 9）。Groves & Thompson 双过程、Frederick & Loewenstein
  享乐适应都指向"重复应递减"。
- 候选：(a) 引入 exposure/新鲜度衰减项；(b) 维持（判为可接受简化）。既有纪要已把此列"未做·可选"，本提案
  请议会正式裁"可选 vs 必改"。

**Q7｜arousal 的双向性——平淡对话该不该主动降唤醒（deactivation）？**
- 现状：arousal 只升不降（整流输入 + 无向低唤醒的回归）。circumplex 下半区（放松/倦怠/平静）在长对话中
  几乎不可达。
- 候选：(a) 给 arousal 独立的向静息/低唤醒回归（对话平淡→降活化）；(b) 维持。
- 与 Q3 联动：Q3 定"静息点在哪"，Q7 定"平淡时会不会主动往那走"。

## 四、建议路由席位

- **数学**（吸引子视角主裁）：把九环画成动力系统、判定 seeking 吸引盆的存在性与稳定性；整流/直流偏置的
  稳态代数、reversion 对零均值 vs 恒正输入的不同表现、习惯化与关系跃迁的动力学形式（标量单调 vs 多稳态）。
- **心理**：Q2（attitude 是否含 arousal 维）+ Q5（关系多稳态）主裁；circumplex 中性点、hedonic adaptation、
  依恋/关系发展是否单调。
- **神经/生物**（可选同席）：自主神经静息唤醒基线、NE 张力、Panksepp seeking 系统语义、habituation 生物学范围。
- **CS**：守红线门——任何"把 LLM/meta 引入 affect 热路径"或"让议会下场定运行时数值"的建议一律 BLOCK；
  确认候选修法（旋钮/公式项/关系维度）不破确定性热路径。

## 五、红线自查（提案侧）

- 本提案**只读**项目 trace 与 `src/` 代码、做确定性根因核验，**未介入情绪/记忆/语言数据的产生**。
- 确定性最小复现用纯 `affect_math` 函数、无 LLM，可复跑；不改任何运行代码、不设默认值。
- 所有候选修法均在「不把 LLM/meta 塞进 affect 热路径」前提内（旋钮 / 纯数学项 / 建构语义调整）。

## 引文

### A. 已核验可复用（2026-06-29 / 2026-06-30 议会现场核验，本提案复用其链接）
- Russell, J. A. (2003). Core affect and the psychological construction of emotion. *Psych. Review* 110(1):145-172. [DOI:10.1037/0033-295X.110.1.145](https://doi.org/10.1037/0033-295X.110.1.145) — 核心情感的个体基线 / affective homeostasis（arousal 亦应有静息锚）。
- Russell, J. A. (1980). A circumplex model of affect. *JPSP* 39(6):1161-1178. [DOI:10.1037/h0077714](https://doi.org/10.1037/h0077714) — valence×arousal 二维环状结构；arousal 维的双极性（deactivation↔activation）。
- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psych. Science* 21(7):984-991. [DOI:10.1177/0956797610372634](https://doi.org/10.1177/0956797610372634) · [PMC2901421](https://pmc.ncbi.nlm.nih.gov/articles/PMC2901421/) — 单调/高惯性漂移=适应不良（arousal 无上限习惯化的病理类比）。
- Sterling, P., & Eyer, J. (1988). Allostasis. · Goldstein & Kopin (2007). *Stress* 10(2):109-120. [PMC4166604](https://pmc.ncbi.nlm.nih.gov/articles/PMC4166604/) — stability through change：应激后应恢复，无上限漂移即病理。
- Groves, P. M., & Thompson, R. F. (1970). Habituation: a dual-process theory. *Psych. Review* 77(5):419-450. [ResearchGate](https://www.researchgate.net/publication/18847090) — 重复同向刺激应衰减（系统缺习惯化 → arousal 不回落）。
- Frederick, S., & Loewenstein, G. (1999). Hedonic adaptation. In *Well-Being*. [条目](https://stafforini.com/works/frederick-1999-hedonic-adaptation/) — 持续刺激下情感强度递减。

### B. 待议会现场核验（方向指引，**主程未亲验 URL，勿臆造**；请各席评审时补稳定标识）
- Kuppens, P., Tuerlinckx, F., Russell, J. A., & Barrett, L. F. (2013). The relation between valence and arousal in subjective experience. *Psychological Bulletin* 139(4). —— arousal 与 |valence| 的 V 形/回旋镖关系，用于裁定 `occ_prior` 的 `0.6·|valence|` 项是否忠实、及中性点 arousal 应取何值。【待核验】
- Bradley, M. M., & Lang, P. J. (1994) / Lang, P. J. (1995). IAPS / motivational priming. —— arousal 维的实证测量、与 valence 的 boomerang 分布。【待核验】
- Barrett, L. F., & Russell, J. A. (1999). The structure of current affect. —— core affect 二维结构与中性参照点。【待核验】
- Panksepp, J. (1998). *Affective Neuroscience*. OUP · Panksepp & Biven (2012). *The Archaeology of Mind*. —— SEEKING 系统（v+,a+ 的探索/渴求动机色彩），用于裁定 `motivational_system` 把该象限判为 seeking 是否忠实、及"seeking 吸引盆"是否对应真实动机神经环路。【待核验】
- 动力系统/吸引子视角（供数学席）：Gottman & Murray 关系动力学、Vallacher & Nowak dynamical social psychology —— 关系是否应建成多吸引子而非标量单调。【待核验】

---
*状态：提案·待 `/science-council` 评审。评审后另立纪要（`2026-07-0x-...-council.md`，遵 `.claude/rules/meeting-notes-citations.md` 引文规则），本文件保留作评审输入存档。*
