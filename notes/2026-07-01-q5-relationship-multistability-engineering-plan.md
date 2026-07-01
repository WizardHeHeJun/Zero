# 工程方案（交回议会）：Q5 关系多稳态——标量 EWMA → 离散关系态

> **本文件是工程师团队交回议会的方案输入（P5），不是纪要、不改 `src/`**。议会裁决（[2026-07-01-seeking-attractor-council.md](2026-07-01-seeking-attractor-council.md) §四 Q5）：关系被建模成标量单调 EWMA 是**失真但非燃眉**——数学席"标量单稳只有唯一不动点、调参无解"、心理席"陌生态本身应是稳定态、EWMA 连'陌生'都不是吸引子只是起点"。CS 席要求**先出工程方案 A/B/C 的耦合估算再回议会选路径**，不直接上完整状态机。本文档给三路径的落点/耦合面/代价/推荐，供议会定语义。

## 一、问题复述（为何调参无解）

`attitude` 是 `tuple[float,float]` 的标量 EWMA（`affect_math.py:attitude_step`）：`a' = (1-rate)·a + rate·s − reversion·(a−setpoint)`。线性一阶差分**全局唯一不动点** `a*=rate·s/(rate+reversion)`，与初值无关、无离散相变。真实关系（Knapp 阶梯 / Altman-Taylor 社会渗透 / 依恋）是**事件门控的离散阶段跃迁**，绝大多数关系稳定停在某阶段、不随时间自动升级。当前引擎缺"保持在陌生态"的负反馈 → P1/P2 压平 arousal 后，valence 侧仍会缓慢单调漂移升温。Q5 是比 P1/P2 更深的结构层。

三路径按"改动面 / 是否进 affect 热路径 / 能否零回归"排序（轻→重）。

---

## 二、路径 A：`familiarity` 门控 `rate` 衰减（最轻，CS 席推荐起点）

**思路**：不引入离散状态，只让"越熟越难改变态度"——熟悉度随交互累计，调低 `attitude_step` 的 `rate`（陌生期态度快成形、熟人期趋稳）。近似"陌生态更稳定"，但仍是单不动点系统（不产生真多稳态）。

**落点（2 文件，纯标量、可零回归）**：
- `chat_driver.py`：复用已落地的 `self.exposure`（P2 已加的曝光计数），派生 `familiarity = 1 − exp(−exposure/τ_f)`；`rate_eff = rate·(1 − k·familiarity)`。
- `affect_math.py`：`attitude_step` 增 `rate` 已是参数（现签名已支持），无需改签名，chat_driver 传 `rate=rate_eff`。
- env：`ZERO_ATTITUDE_RATE_DECAY_K`（默认 0=不衰减，零回归）、`ZERO_FAMILIARITY_TAU`。

**耦合面**：极小。无 state/记忆/持久化改动（exposure 已在 chat_driver 本地、会话内）。不进 affect 热路径的 LLM 分支。
**代价（数学席）**：仍是**单不动点**系统——只是让漂移变慢，不能表征"陌生↔朋友"两个独立稳态、也无事件门控跃迁。治标不治本，但成本近乎零、可立即验证"减缓升温"效果。
**跨会话**：exposure 当前仅会话内（重启归零）。要真"越来越熟"需持久化 familiarity（见路径 C 的记忆接线，属中等成本）。

---

## 三、路径 B：`familiarity` 仅作 LLM 提示层"关系距离"标签（不进热路径）

**思路**：affect 数学完全不动；把 familiarity（由 `attitude_v` 或 exposure 单调映射）离散成"初识/相熟/熟络"标签，仅注入 `_CONVERSE_SYS`/人设卡作**关系距离语境约束**，让 LLM 自己据此把握分寸（"你们才刚认识，别骤然亲密"）。直接对治"seeking→暧昧"里 **LLM 层的语义漂移**（议会张力一裁定的独立失真）。

**落点**：
- `state.py`/`chat_driver.py`：派生 `familiarity_label`（纯字符串标签，不进 affect 数学）。
- `language_openai.py` `_CONVERSE_SYS`：注入关系距离一句（空/默认=旧 prompt，零回归）。
- env：`ZERO_RELATIONSHIP_STAGE_HINT`（默认关）。

**耦合面**：小，且**全在 LLM 提示层**——不碰 `affect_math`、不碰 state 持久化。
**代价**：不改变引擎动力学（attitude 仍单调），只给 LLM 一个"刹车提示"；效果依赖 LLM 是否遵从提示（软约束）。**守红线**：这是"给 LLM 上下文"而非"把 LLM 塞进 affect 热路径"，合规；但需议会确认"关系标签由 attitude 单调派生"是否够用，还是必须真离散态（路径 C）。
**互补**：B 与 A 正交，可叠加（A 减缓引擎漂移 + B 给 LLM 距离锚）。

---

## 四、路径 C：真离散关系态 + 事件门控跃迁（完整，最重）

**思路**：引入与情感 (v,a) **分离**的关系维度（如 `relationship_stage: stranger|acquaintance|friend|close` 或连续 `intimacy∈[0,1]` 配双稳动力学），跃迁由**真实事件门控**（非时间函数）。数学席给两种建模：
- **C1 离散状态机**：阶段 + 阈值事件触发跃迁（升级需显著正事件累积、降级需冲突），每阶段是稳定态。
- **C2 tanh 双稳**：复用本仓库 `mood_step`（`affect_math.py:mood_step`）的 pitchfork 结构（`self_gain·self_k > 1−inertia` 即双稳），令关系维度有"陌生盆/亲密盆"两个吸引子 + 中间不稳定鞍点，跨越需越过势垒（对应"关系需要投入换取"）。

**落点（跨层，改动面大）**：
- `state.py`：`AffectState` 新增关系字段 → **Checkpointer 序列化格式变**（旧 checkpoint 需默认值兜底 / 迁移；参 `runner.py:ALLOWED_CHECKPOINT_TYPES` 白名单坑）。
- `affect_math.py`：新增 `relationship_step`（C2 复用 `mood_step` 数学）或状态机纯函数。
- `chat_driver.py`：跃迁事件判定（确定性标量门控，**不得**用 LLM 判"是否该升级"——那条 CS 席 **BLOCK**）。
- `src/memory/`：关系态**跨会话持久化** → 触发 memory-rules（任务完成节点写、显式 `user` scope、不每步写）。这是最重的耦合。
- `persona.py`：L3 预置关系（`initial_attitude`/`seed_memories`）与运行时关系态机对齐。

**耦合面**：大（state + 热路径 + 记忆层 + persona 五处）。breaking 风险在 Checkpointer 序列化与记忆持久化。
**代价**：真正解决 Q5（多稳态 + 事件门控），但工程成本与回归面最高。
**红线**：跃迁门控必须确定性（标量事件），跃迁语义（几个阶段、升/降级条件、双稳参数）属 **[议会定语义]**，工程师不得私定。

---

## 五、耦合估算汇总 + 推荐

| 路径 | 改动文件 | 进热路径? | 记忆层? | Checkpointer breaking? | 零回归 | 治本度 |
|---|---|---|---|---|---|---|
| **A** rate 衰减 | 2（chat_driver, affect_math 传参） | 否 | 否（会话内） | 否 | ✅ env 默认 0 | 低（仍单不动点） |
| **B** LLM 距离标签 | 2（state 派生, _CONVERSE_SYS） | 否（LLM 提示层） | 否 | 否 | ✅ 默认关 | 中（治 LLM 漂移，不改引擎） |
| **C** 离散/双稳关系态 | 5（state/affect_math/chat_driver/memory/persona） | 门控标量（否 LLM） | 是（跨会话） | 是（需迁移） | 需设计默认关 | 高（真多稳态） |

**工程师团队推荐（供议会裁）**：**分阶段 A+B 先行、C 立项**。
1. **先做 A+B（低成本、正交、均零回归）**：A 减缓引擎单调漂移、B 给 LLM 关系距离锚——组合即可显著缓解"内容无关升温"，且不碰记忆层/序列化。作为 P1/P2 之后的**第二轮止血**。
2. **C 作为独立里程碑立项**：需议会先定语义（阶段划分 vs 连续双稳、跃迁事件的确定性判据、C2 的 `self_gain/self_k/inertia` 参数范围），并出 Checkpointer/记忆迁移方案，再交工程实现。**不与 A/B 并行赶工**。

**回议会待定语义**：
- Q5-α：关系用**离散阶段**（C1）还是**连续双稳**（C2，复用 mood_step pitchfork）？
- Q5-β：跃迁的**确定性事件判据**是什么（正事件累计阈值？冲突降级？时间不触发升级如何保证）？
- Q5-γ：关系态是否必须**跨会话持久化**（C 的记忆耦合来源），还是会话内即可（A 的轻量近似够不够）？
- Q5-δ：与 P1-b 的 `attitude`（valence 维长期评价）如何分工——关系维度（熟悉/信任/亲密）与 attitude(valence) 是否正交？

---

## 引文（复用议会纪要现场核验条目）

- Knapp, M. L. (1978). *Social Intercourse: From Greeting to Goodbye*. — 关系发展离散阶段、需事件触发不自动升级。[Wikipedia](https://en.wikipedia.org/wiki/Knapp%27s_relational_development_model)
- Altman, I., & Taylor, D. A. (1973). *Social Penetration Theory*. — 亲密度离散层次、可在任意阶段稳定/退化。[communicationstudies.com](https://www.communicationstudies.com/communication-theories/social-penetration-theory)
- Gottman, J. M., & Murray, J. D. (2002). *The Mathematics of Marriage: Dynamic Nonlinear Models*. MIT Press. — 关系动力学多吸引子（双稳不动点以鞍点分离），双稳来自非线性 influence 函数、非 EWMA。[MIT Press](https://direct.mit.edu/books/monograph/2547/The-Mathematics-of-MarriageDynamic-Nonlinear) · [ResearchGate 232424148](https://www.researchgate.net/publication/232424148)
- Vallacher, R. R., & Nowak, A. (1994). *Dynamical Systems in Social Psychology*. — 社会关系的吸引子/自组织框架。[research page](https://psy2.fau.edu/~vallacher/research_DSP.html)
- Strogatz, S. H. *Nonlinear Dynamics and Chaos*. — pitchfork 分叉/双稳临界条件（C2 参数范围依据）。[作者页](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)

---
*工程师团队方案输入，2026-07-01。不改 `src/`；A/B/C 均在"不把 LLM/meta 塞进 affect 热路径"红线内（C 的跃迁门控为确定性标量）。待议会定 Q5-α/β/γ/δ 语义后再实现。*
