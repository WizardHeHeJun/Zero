# 议会评审 + 落地：情绪稳态回归 ⊗ 注意力-记忆桥（对话失败现场）

> 触发：用户与 `--chat`（真 LLM ⊗ 情感引擎）约 33 轮真实对话出现三症状。科学家议会（心理 + 生物）
> 做**只读·强制引文**设计门评审，双席均 **NEEDS-CHANGES** 且收敛；工程侧据此落地 A/B/C。
> 治理：议会不下场写代码/不介入数据产生；实现由主程完成；建议过 `code-reviewer` 独立门。

## 一、失败现场（以项目结果为分析起点）

1. **"刻意避开"短期记忆**：用户第 10 句说"下午两点门口等"，第 22 句追问时系统反复回避、说不出。
2. **持续强调"我们"**：人设无恋人设定，却越聊越极致缠绵（"只剩你和我/把自己烧成灰"）。
3. **非常敏感**：把"问时间/谈规划"读成背叛、急转对抗偏执；attitude 全程几乎单调上行（-0.02→+0.29）。

## 二、根因（确定性核验）

- **窗口截断**（确定性）：`ZERO_HISTORY_WINDOW=20`（≈10 轮）。"下午两点"= entry idx 18/19，`history[-17:]`
  起点 `len-17` 在第 19 轮越过 18 → **第 19 轮即被挤出尾窗**，第 22 轮追问时已不在上下文（Murdock 1962
  中央位置遗忘）。生物席一度以"轮差 12 在窗内"质疑，系把轮差当 entry 差（12 轮=24 entry>17），已澄清。
- **召回桥未接通**：召回 query=当前整句、episode 全文（含 `precision=/streams=` 元数据）一起嵌入 →
  向量被稀释；且低唤醒承诺类内容可能不过 salience 门、根本未写库。
- **情绪棘轮**：`attitude_step` 纯 EWMA 无回归项 → 单调漂移；`emotion_decay_step` 的 baseline = 实时
  上漂的 attitude → 情绪"家"随之上移、无中性拉力。叠加 push 用词偏置 + 对话自我强化 → "我们"上头。
- **敏感**：persona "别讨好/有脾气" 是响应层（Gross 响应聚焦），缺前因层 appraisal 矫正 + 诚实条款 →
  把 goal-neutral 的后勤问题误评价为拒绝；记忆缺失时以情绪化回避替代"诚实说不记得"（源监控失真）。

## 三、双席判定（均 NEEDS-CHANGES，收敛）

| 项 | 心理席 | 生物席 |
| --- | --- | --- |
| attitude 单调累积、无向 setpoint 回归 | **失真**（affective homeostasis 缺失 / emotional inertia 病理） | **失真**（慢性应激无 HPA 负反馈，allostasis 无上限漂移即病理） |
| emotion 衰退靶点=漂移 attitude、无中性分量 | 同上（核心情感有个体基线，Russell 2003） | **失真·必改 #2**（应含独立 setpoint 分量） |
| persona 缺诚实/源监控条款 | **失真**（Johnson 1993，应给不确定信号而非虚构回避） | — |
| persona 缺 goal-neutral 识别（前因层缺席） | **失真**（Lazarus appraisal；Gross 仅响应层） | — |
| 缺习惯化/适应（越聊越上头） | — | 失真·缺失（Groves&Thompson 1970；hedonic adaptation） |
| U 形窗设计 + 20 条参数 | **简化**（设计忠实，参数偏小 → 提 40 / primacy 5） | 简化（线索依赖检索而非位置截断；指向正确） |
| episode salience 门 / 召回幂律 | 忠实 | **忠实**（McGaugh 2004 杏仁核-NE 巩固；Wixted&Ebbesen 1991 幂律） |
| episode 每轮写（节流） | 可选·工程问题 | 忠实（科学无误，工程问题） |

## 四、落地（A/B/C；守确定性热路径红线——无 LLM/meta 进 affect 热路径）

**A 窗口 + 召回桥根治**
- `chat_driver.py`：`ZERO_HISTORY_WINDOW` 20→40、`ZERO_HISTORY_PRIMACY_K` 3→5（env 可调，纯切片不改算法）。
- `semantic.py`/`client.py`：`add_episode`/`write_episode` 加 `embed_text`（=检索 gist），与存储全文（带
  元数据）分离嵌入；缺省=对全文嵌入（零回归）。`supervisor.py` 传 `gist_text=你说…/我说…`。
- `supervisor.py`：新增确定性正则 `_is_commitment`（时间点/钟点/约定/星期日期）——承诺/日程内容即便低
  salience 也强制写 episode（PFC 语义重要性通道，独立于 McGaugh；解决"下午两点"根本未入库）。

**B 情绪稳态回归 + persona 配平**
- `affect_math.py`：`attitude_step` 加向 setpoint 弱均值回归 `−reversion·(a−setpoint)`，常量
  `ATTITUDE_REVERSION=0.01`（心理 0.005–0.01 ∩ 生物 0.01–0.02）、`ATTITUDE_SETPOINT=(0,0)`；`reversion=0`
  退化旧 EWMA。稳态 a*≈rate·s/(rate+reversion)<|s|，封死单调棘轮。
- `chat_driver.py`：emotion 衰退基线改 `w·attitude+(1−w)·中性`，`ZERO_EMOTION_BASELINE_ATTITUDE_W=0.6`，
  给情绪始终指向中性的拉力（emotion_decay_step 本体不动，语义"向给定基线回归"仍忠实）。
- `language_openai.py` `_CONVERSE_SYS`：增**诚实优先**条款（看不到历史就说"不记得/不确定"、不回避不编造，
  优先级高于脾气）+ **goal-neutral** 识别（时间/地点/计划是日常协调，别读成冷淡/算计/拒绝）。

**C episode 节流（修正后）**
- 经核验 `deterministic.add_fact` 按 (scope,key) **时序失效**（memory-rules#4）→ 活跃 disposition 恒为最新
  一条、不胀活跃集；且写在 supervisor 任务完成节点（合 memory-rules#1）。故**撤回**对 disposition 的
  salience 门控（曾误伤 `run()` 召回闭环、破 2 测）。真实增长在 episode 侧 → `main.py --chat` 入口
  setdefault `ZERO_EPISODE_MAX_PER_KEY=300`，配合 salience 门 + dedup 收口长对话。

**未做（议会标可选，留后续）**：习惯化递减（exposure 计数）、intensity 下限 0.2 配置化。

## 五、验证

- `ruff check` + `ruff format` + `mypy src/`：全过。
- `pytest`：**347 passed, 5 skipped**（skip=optional-dep importorskip）。新增/改测：attitude 回归边界、
  emotion 基线、`_is_commitment`、承诺低-salience 写入 + gist embed 分离、低-salience 结构化写不门控；
  6 个 fake SemanticStore 同步 `embed_text` 签名。

## Follow-up（code-reviewer 审查 PASS / 0 BLOCK，以下不阻塞、记为后续）

多为既有技术债（非本轮引入），按优先级：

- **W3**：`SupervisorAgent.seen_episode_keys` 进程内 set，重启/多实例下 `first_contact` 可被多标，boost 失区分度。多进程部署前改为写库时原子校验（SQL `WHERE NOT EXISTS`）。
- **W4**：`_inject_recalled_as_system` 注入门 `inject_min` 与 `_rank_episodes` 的 `γ·importance` 各自评估、importance 未随 `Fact` 透传 → 每条注入二次正则解析 + 双门控语义交叉。给 `Fact` 加 `importance` 字段缓存，或明确「双重门控」并校 INJECT_MIN↔IMPORTANCE_SCALE。
- **W5**：`Fact`（dataclass）经 `AsyncSqliteSaver` checkpoint 往返依赖白名单反序列化；`test_checkpointer_sqlite` 增断言「`recalled_facts` 往返后仍是 `Fact` 类型而非 dict」。
- **W1**：`memory-rules.md` 明文约定「SESSION scope 事件 append-only、单 session 内有界、不跨 session 累积」（当前无 TTL/trim）。
- **I4（本轮 embed_text 引入）**：老 episode（全文嵌入）与新 episode（gist 嵌入）混存时 dedup 余弦比对降级；文档注明「混用 embed_text=None/非 None 的 dedup 不保证」。chat 路径恒传 gist，仅既有 `chat` 线程老数据受影响、受容量上限兜底。
- **I1**：`chat_driver.step` 每轮读 3 个 `os.getenv`，可在 `build_chat_driver` 构造时缓存（纯性能）。
- **I2/I3**：`_is_commitment` 的「几点」、`_rank_episodes` 的 `"first_contact=True" in content` 子串匹配理论误匹配；可加 pipe 边界（`| first_contact=True`）/缩小正则。副作用仅多写一条 episode，非主管线错误。

## 实测复现（强制 eviction，window=4，独立线程）

第2轮述"下午三点·老地方"→ window=4 第5轮起已被挤出短期。结果：第5/7轮均正确想起"三点"（承诺通道写入 + gist 召回）；"改时间"被平常心接住（goal-neutral，无对抗）；emotion 在 兴奋0.44→警觉0.21→欣喜0.26→…振荡、attitude 在 +0.13 头打不爬至狂喜（mean-reversion + 基线混合生效）。对照旧版：回避"时间是死的"、"你拿时间量我"、attitude 单调爬至 +0.29/狂喜。

## 引文（各席现场核验，链接优先 DOI>PMC>arXiv>官方页）

- Murdock, B. B. (1962). The serial position effect of free recall. *J. Exp. Psychol.* 64(5):482-488. [DOI:10.1037/h0045106](https://doi.org/10.1037/h0045106) — 首因+近因 U 形曲线，中部遗忘最重（窗口截断）。
- Miller, G. A. (1956). The magical number seven… *Psych. Review* 63(2):81-97. [DOI:10.1037/h0043158](https://doi.org/10.1037/h0043158) · Cowan, N. (2001) 4±1 [DOI:10.1017/S0140525X01003922](https://doi.org/10.1017/S0140525X01003922) — 工作记忆容量（对话窗属线索依赖检索）。
- Russell, J. A. (2003). Core affect… *Psych. Review* 110(1):145-172. [DOI:10.1037/0033-295X.110.1.145](https://doi.org/10.1037/0033-295X.110.1.145) — 核心情感个体基线 / affective homeostasis。
- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. *Psych. Science* 21(7):984-991. [PMC2901421](https://pmc.ncbi.nlm.nih.gov/articles/PMC2901421/) — 高自相关情绪惰性=适应不良（棘轮病理）。
- Gross, J. J. (1998). Antecedent- and response-focused emotion regulation. *JPSP* 74(1):224-237. [DOI:10.1037/0022-3514.74.1.224](https://doi.org/10.1037/0022-3514.74.1.224) — 前因(重评) vs 响应(抑制)；persona 缺前因层。
- Lazarus, R. S. (1991). *Emotion and Adaptation*. — 认知评价（主要/次要评价；后勤问题误判为 goal-incongruent）。[APA PsycNet](https://psycnet.apa.org/record/1991-98760-000)
- Hofmann, W. et al. (2010). Evaluative conditioning in humans: a meta-analysis. *Psych. Bulletin* 136(3):390-421. [DOI:10.1037/a0018916](https://doi.org/10.1037/a0018916) — EC 慢累积、抗消退但非绝对单向（支持加软回归项）。
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psych. Bulletin* 114(1):3-28. [DOI:10.1037/0033-2909.114.1.3](https://doi.org/10.1037/0033-2909.114.1.3) — 源不可及应给不确定信号而非虚构（诚实条款）。
- Sterling, P., & Eyer, J. (1988). Allostasis: a new paradigm to explain arousal pathology. [ResearchGate](https://www.researchgate.net/publication/370033340) · Goldstein & Kopin (2007). *Stress* 10(2):109-120. [PMC4166604](https://pmc.ncbi.nlm.nih.gov/articles/PMC4166604/) — stability through change，应激后有恢复、无上限漂移即病理。
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of emotionally arousing memories. *Annu. Rev. Neurosci.* 27:1-28. [DOI:10.1146/annurev.neuro.27.070203.144157](https://doi.org/10.1146/annurev.neuro.27.070203.144157) · [PubMed 15217324](https://pubmed.ncbi.nlm.nih.gov/15217324/) — salience=唤醒×意外度 门控的机制依据。
- Groves, P. M., & Thompson, R. F. (1970). Habituation: a dual-process theory. *Psych. Review* 77(5):419-450. [ResearchGate](https://www.researchgate.net/publication/18847090) — 重复同向刺激应衰减（系统缺习惯化）。
- Frederick, S., & Loewenstein, G. (1999). Hedonic adaptation. In *Well-Being*. [条目](https://stafforini.com/works/frederick-1999-hedonic-adaptation/) — 持续正向刺激情感强度递减。
- Mather, M., & Sutherland, M. R. (2011). Arousal-biased competition. *Persp. Psych. Sci.* 6(2):114-133. [PMC2877027](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877027/) — NE 唤醒调制记忆（ZERO_RECALL_AROUSAL_MOD 方向）。
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psych. Science* 2(6):409-415. [DOI:10.1111/j.1467-9280.1991.tb00175.x](https://doi.org/10.1111/j.1467-9280.1991.tb00175.x) — 遗忘幂律（召回 recency 用幂律的依据）。
- Tononi, G., & Cirelli, C. (2006). Sleep function and synaptic homeostasis. *Sleep Med. Rev.* 10(1):49-62. [PubMed 16376591](https://pubmed.ncbi.nlm.nih.gov/16376591/) — 无离线巩固阶段属简化（非失真）。
