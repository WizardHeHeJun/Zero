# 短时注意力↔长时记忆桥设计决策（科学家议会综合 · 2026-06-29）

- 日期：2026-06-29
- 议题：补齐情感引擎⊗LLM仿生人「短时注意力↔长时记忆」之间的桥——短时端 `history[-20:]` 死窗口、召回「拼字符串」未进注意力预算、召回线索单一无三维打分。三解法：A 有 token 预算的注意力装配器 / B `recall` 之上三维重排(recency×salience×relevance) / C 遗忘=时序失效+容量上限不物理删。
- 评审席位：神经（CLS/海马↔皮层巩固/显著网络）· 心理（工作记忆容量/系列位置/情景语义/遗忘曲线）· 数学（检索打分/衰减函数/背包/double counting）· 计算机（守红线·必到）；主持席综合。生物席弱相关（记忆架构非生理），控成本跳过。
- 接续：上一轮 `notes/2026-06-25-conversation-episodic-memory-council.md` 明确把「完整巩固/遗忘曲线」列为**路线图后期**，本轮即接此线评审。
- 治理：议会只读、强制现场引文、不改代码、不介入数据产生。本文件是唯一写动作。
- **总判定：NEEDS-CHANGES**（8 条必改，含心理席唯一纯失真 primacy、数学席乘积→加权和+幂律、CS 席 4 条 BLOCK）。

## 各席判定汇总

| 要素 | 神经 | 心理 | 数学 | CS |
| --- | --- | --- | --- | --- |
| `history[-20:]` 硬截断 | 简化 | 简化（20 条远超 Cowan 注意焦点≈4 chunks） | 简化 | 简化（须 fallback） |
| `recalled_str` 拼字符串旁路 | **失真**（绕过 attention 预算竞争） | 简化 | 简化 | **BLOCK**（缺空集 fallback） |
| salience 写入门控 | 忠实（DA-NE 门控+SWR 偏置） | **失真**（漏低唤醒高语义内容） | 简化（可防守，=精度加权 PE） | 忠实（任务完成节点） |
| salience 检索复用 | 忠实（杏仁核调制检索） | 忠实 | **失真**（写入显著≠检索相关，类比 BM25 把 IDF 当 TF） | 忠实（读侧无写副作用） |
| 衰减函数（指数） | 偏失真（时间细胞支持幂律） | **失真**（Ebbinghaus 幂律） | **失真**（Wixted&Ebbesen 实证幂律 t⁻ᵈ） | 简化（应参数化） |
| primacy 缺失 | 不评 | **失真（唯一纯失真）**（Murdock U 形） | 不评 | 简化（加 K 条头部，成本低） |
| 三维乘积合成 | 忠实（三维分工对应神经机制） | 忠实 | **失真**（任一维零则归零，应加权和） | 忠实（确定性计算） |
| 频率项缺失（ACT-R） | 不评 | 简化 | 简化→需补（反复提到的话题更易召回） | 简化（后期补） |
| salience 与 rpe 双计 | 不评 | 不评 | **失真风险**（salience 已含 \|rpe\|，平方放大） | 防御性注释 |
| 情景/语义划分(write_episode/disposition) | 忠实（CLS） | 忠实（Tulving） | — | — |
| 遗忘=时序失效不物删 | 忠实（Hardt 主动衰退/再巩固） | 忠实（方向对） | 忠实（幂律实现形式） | 忠实（invalid_at 已实现） |
| LLM 进装配/重排热路径 | — | — | — | **BLOCK** |
| 读节点写副作用 | — | — | — | **BLOCK** |
| embedding 向量进 state | — | — | — | **BLOCK** |

## 跨学科张力裁决

- **张力①｜salience 双重角色（神经「忠实」vs 数学「检索复用失真」）**：两席均对、角度不同。神经依 McGaugh 2004 / Ranganath&Ritchey 2012——杏仁核确实调制检索、情绪显著记忆更易召回；数学指出写入门控的 salience（值不值得记）与检索的 relevance（现在该不该取）是两个语义。**裁决**：salience 保留为打分中**独立的 importance 先验维度**（γ·importance），不与 relevance 混为一维、不与 rpe 双计；神经建议的「查询时乘当前唤醒作 NE 调制代理」实现为可选 `γ_eff = γ·(1+0.5·arousal)`。
- **张力②｜乘积 vs 加权和**：数学判乘积**失真**（任一维趋零则得分归零，过度惩罚「久远但重要」「语义偏但高唤醒」），ACT-R 是对数相加结构；余三席无异议。**裁决**：采**加权和**。
- **张力③｜指数 vs 幂律**：数学(Wixted&Ebbesen 1991/1997)+心理(Murre&Dros 2015)双判幂律忠实、指数失真；神经(Howard&Eichenbaum 时间背景漂移)亦近幂律。**裁决**：衰减项用 **Δt⁻ᵈ（幂律）**，d 走 env 默认 0.5。
- **张力④｜primacy 漏项**：心理唯一纯失真(Murdock 1962，记忆 U 形首尾皆强)，现有代码零缓解；他席不评亦不反对。**裁决**：**必补 U 形**——`history[:K]+history[-(N-K):]`，并对首轮 episode 打 `first_contact` 元标签、召回上调系数。

## 决策（逐条可落地·标注归属/确定性/零回归/默认）

> 全部确定性、无 LLM 进热路径、以 env 开关保零回归。落地后须 code-reviewer 独立审查。

### D1 `recalled_str` 并入 history 预算竞争（神经失真 + CS BLOCK-B）
- 归属：`src/orchestration/chat_driver.py` `ChatDriver.step()`。
- 改：高 importance episode（超 `ZERO_RECALL_INJECT_MIN`，默认 0.25）以 `{"role":"system","content":"（记忆片段）…"}` 插入 `history` 头进入同一预算窗竞争；其余仍走 `recalled_str` 低优先背景。
- **fallback（硬约束）**：`recalled==[]` 时退化为 `history[-20:]`，无 index 越界。确定性✓ 零回归✓（无 episode 行为不变）。

### D2 history 改 U 形切片（心理唯一失真·primacy）
- 归属：`chat_driver.py` `step()` 的 `self.history[-20:]` 处。
- 改：`self.history[:K] + self.history[-(N-K):]`，K=`ZERO_HISTORY_PRIMACY_K`(默认 3)、N=`ZERO_HISTORY_WINDOW`(默认 20)；`len(history)<=N` 不截；**K=0 退化为现状**（零回归开关）。建议默认开（修失真非新功能）。

### D3 三维重排：加权和 + 幂律（数学失真修复合并）
- 归属：`src/orchestration/memory_recall.py`（新辅助 `_rank_episodes()` 或 `MemoryRecallAgent.__call__` 内）。
- 公式（加权和，非乘积）：
  `score_i = α·Δt_i⁻ᵈ + β·sim_i + γ·importance_i`
  - `Δt_i` = `(now - fact.valid_at)` 折算**天**，下界 clamp 1.0（防溢出）；`d`=`ZERO_RECALL_DECAY_D` 默认 0.5（幂律，Wixted&Ebbesen）。
  - `sim_i` = 向量余弦，来自 D4（`StoredFact.sim` 标量透传）。
  - `importance_i` = 写入时 salience，从 episode 文本正则解析 `precision=…`（`supervisor.py` 已固化；**禁调 LLM 打分**）。
  - `α/β/γ` 走 env `ZERO_RECALL_ALPHA/BETA/GAMMA`，默认 0.33/0.34/0.33（占位、待实跑校准）；设 0/0/1 退化为旧行为。
  - 可选唤醒调制：`γ_eff = γ·(1+0.5·clamp(arousal,0,1))`，arousal 取 `state.affect_sample[1]`。
- **double counting 防御**：importance 已含 `|rpe|`，**禁**将 `state.rpe` 作第四独立维再加入公式。确定性✓ 零回归✓。

### D4 `search` 返回携带 sim 分数（D3 前置依赖）
- 归属：`src/storage/backends/semantic.py` `SqliteVectorStore.search` 及 `SemanticStore` 协议、`src/memory/client.py` `MemoryClient.recall`。
- 改：`StoredFact` 加可选 `sim: float = 0.0`；sqlite_vec 填真值、Graphiti 填 0.0；`recall` 透传。
- **CS BLOCK-D 防御**：只透传**标量 sim**，embedding 向量（`list[float]`）**禁进** `Fact`/`AffectState`，用完即弃。确定性✓ 零回归✓（新增可选字段不破坏 `query_facts`）。

### D5 first_contact primacy 元标签（D2 写侧对应）
- 归属：`src/orchestration/supervisor.py` `SupervisorAgent.__call__()`。
- 当且仅当本轮为 session 首轮：episode_content 追加 ` | first_contact=True`（判据：写前查同 session episode 数为 0，或轮计数器）。
- 召回侧 D3：`importance_i *= 1.2 if "first_contact=True" in fact.content else 1.0`。老 episode 无字段系数=1.0。确定性✓ 零回归✓。

### D6 salience 门控补低唤醒高语义豁免（心理失真·注释级）
- 归属：`supervisor.py:53` `ep_threshold` 逻辑后。
- 注释标注「低唤醒高语义重要内容（如用户平静自我披露『我在换工作』）可能被 `ZERO_EPISODE_SALIENCE_MIN`(默认0.15)过度门控漏写，调低/置 0 可缓解」。可选精细修复：`salience_eff = salience + 0.3·abs(value)`（默认不开）。确定性✓ 零回归✓。

### D7 遗忘=幂律时序失效 + 容量上限（数学+心理失真，Hardt 佐证）
- 时间权重：D3 的 `Δt⁻ᵈ` 已在 recall 侧实现；`search` 只需正确返回 `valid_at`（已有），**无需改 search 本身**。
- 容量剪裁：`ZERO_EPISODE_MAX_PER_KEY`（默认 0=不限）在 `SqliteVectorStore.add_episode` **写后**删最旧超量 episode（按 `valid_at asc`）——属容量管理非遗忘建模，**须经 Supervisor 任务完成节点**触发（CS BLOCK-C：禁读节点/热路径触发写、禁每轮全量扫描）。确定性✓ 零回归✓。

### CS 席 4 条 BLOCK（硬约束·进 /engineer 前须排除）
1. **热路径**：装配/重排公式纯数值，importance 用正则解析；**禁** MemGPT 式「LLM 决定 page-in/out」或「LLM 给 fact 评分」（违 CLAUDE.md 确定性热路径）。
2. **无 fallback**：D1 `recalled==[]` 必退化 `history[-20:]`。
3. **读节点写副作用**：失效/剪裁只在 SupervisorAgent（任务完成节点）经 MemoryClient；MemoryRecallAgent/chat_driver 主流程禁写。
4. **向量进 state**：MMR/sim 只在存储层内用，`AffectState` 禁出现 `list[float]` 向量字段（state 不放大对象）。

## 可接受的刻意简化（及代价）
- history N=20 远超 Cowan 4 chunks：LLM 需多轮上下文才连贯，token 预算由 LLM 侧管，可接受。
- **频率项缺失**（ACT-R `ln(Σtₖ⁻ᵈ)`）：需为每 episode 维护访问计数，本轮不补——反复出现的话题不能额外加权，列下一期议会议题。
- salience 门控 `precision×|rpe|` 粗糙：高 precision+低 rpe 的「预期内重要事件」可能漏写。
- 无完整睡眠巩固/离线批处理遗忘：本轮只做幂律召回权重 + 容量剪裁，完整巩固/遗忘曲线仍留后期。
- MMR diversity 维度本轮不强制（sqlite_vec 已有 0.92 dedup 部分覆盖）。

## 必须改（失真 / BLOCK 清单）
1. `recalled_str` 旁路绕过注意力预算（D1）——神经失真 + CS BLOCK-B。
2. primacy 缺失（D2+D5）——心理唯一纯失真，Murdock 1962。
3. 重排改加权和（D3）——数学失真，乘积过度惩罚。
4. 衰减改幂律 `Δt⁻ᵈ`（D3）——数学+心理双判失真。
5. salience 检索角色与 relevance 显式分离（D3 importance 独立维）——数学失真。
6. `search` 返回带 sim（D4）——D3 前置依赖。
7. rpe 不作独立第四维（D3 约束）——数学 double counting。
8. 时间权重进召回（D3 `Δt` 项）——神经+心理+数学三席均指缺失。

## 悬而未决（工程落地时定）
1. `α/β/γ` 默认权重（等权 0.33 为占位，需实跑对话校准）。
2. `ZERO_RECALL_INJECT_MIN`（升入 history system 条目阈值）默认值。
3. 唤醒调制 `γ_eff` 是否默认开（神经建议，本项目尚无实验数据）。
4. first_contact 首轮判据（轮计数 vs 查 episode 数；轮计数更轻但需 AffectState 加计数字段）。
5. D6 低唤醒豁免是否工程化（注释级 vs 纳入 `salience_eff`）。
6. 频率项补全时机（ACT-R 完整形式，建议下一期议会）。

## D8 — importance 归一化（落地后真后端 dogfood 触发的数学席专评 · 2026-06-29）

**触发**：PR #38 落地后跑真后端 smoke（qwen-flash + gemini embedding）暴露结构性问题：`importance` 源 = 写入的 `affect_precision`，而它是 `0.5·(1/σ²)`（方差倒数）**天然无界**（实测 28–72）。后果：三维 `score=α·Δt^(-d)+β·sim+γ·importance` 里 `γ·importance≈24` 碾压 `recency/sim`（≤0.34），退化为「只按 precision 排序」；`INJECT_MIN=0.5` 形同虚设。这是上面「悬而未决 #1/#2」实跑后的实证。

**评审**：聚焦 mini-review，数学席（主）+ CS 席（红线必到），两席无张力、判定一致。

| 子问题 | 判定 | 决策 |
| --- | --- | --- |
| 当前直接用无界 precision | **失真**（量纲失配，三维退化为单维） | 必改 |
| 归一函数形式 | Hill 饱和 `p/(p+C)` 忠实（与 Kalman 增益/逆方差加权同构、有界单调、边际递减） | 采用，优于 min-max/softmax（集合依赖）、exp |
| scale C | 固定 env 旋钮忠实；**自适应集合统计=BLOCK**（破坏可复现/确定性，CS 红线） | `ZERO_RECALL_IMPORTANCE_SCALE` 默认 30（匹配实测量级，使 INJECT_MIN=0.5 成「高质量门」） |
| 落点 | 读侧 `_rank_episodes` + 注入门两处一致归一（不改 episode 写入，免版本迁移） | `normalized_importance()` 公开纯函数；`Fact.content` 不动 |
| 归一后 α/β/γ 等权 | 简化（可接受，量纲对齐后等权是合理起点） | 维持 0.33/0.34/0.33 占位待 dogfood 校准 |
| importance 源 precision vs salience(×\|rpe\|) | 规范偏差（D3 原文 salience，实现用 precision；二者均可辩护、都无界） | 维持 precision（已归一），记录偏差，未来可回议会统一 |

**落地**：`normalized_importance(content)=p/(p+C)`（memory_recall.py），`_rank_episodes` 与 `chat_driver._inject_recalled_as_system` 一致调用；附 `ZERO_RECALL_IMPORTANCE_SCALE` env。另修 WARN-4 余漏：`Scope` 也入 `ALLOWED_CHECKPOINT_TYPES`（Fact.scope 反序列化）。复跑 smoke 确认：`precision=72.8→0.71`、`37.1→0.55` 均归一入 [0,1]、门控恢复区分；Scope 告警消失。新增测试：归一有界单调 + domination 修复（近+相关压过纯高精度）。`pytest` 333 passed / 5 skipped。

### D8 引文（数学席现场核验）

- ACT-R base-level activation：Anderson, J. R. (2021). Foundation of Base-Level Activation. [act-r.psy.cmu.edu PDF](http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2021/07/ACTR2021anderson.pdf) · [Anderson & Bothell et al. (2004) An Integrated Theory of the Mind PDF](http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/526FSQUERY.pdf) — importance↔源激活而非 BLA 频率项。
- 逆方差加权 / Kalman 增益天然有界 [0,1]：[Inverse-variance weighting (Wikipedia)](https://en.wikipedia.org/wiki/Inverse-variance_weighting)。
- 最优线索融合（精度加权权重归一总和=1）：Ernst, M. O. & Banks, M. S. (2002). *Nature* 415:429-431. [nature.com/articles/415429a](https://www.nature.com/articles/415429a)。
- 三维归一后等权合并先例：Park et al. (2023). Generative Agents. [arXiv:2304.03442](https://ar5iv.labs.arxiv.org/html/2304.03442) — importance 先限 [1,10] 再 min-max；本项目 precision 无界故用 Hill 更鲁棒。
- 精度作 softmax 温度（无界精度作权重须有归一分母）：Parr, T. & Friston, K. J. (2017). *Sci. Rep.* 7:14678. [DOI:10.1038/s41598-017-15249-0](https://doi.org/10.1038/s41598-017-15249-0)。

## 引文（各席现场核验）

**神经科学席**
- McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychol. Rev.* 102(3):419-457. [DOI:10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419) — CLS 框架（情景/语义双系统）。
- O'Reilly, R. C., Bhattacharyya, R., Howard, M. D. & Ketz, N. (2014). Complementary learning systems. *Cogn. Sci.* 38(6):1229-1248. [DOI:10.1111/j.1551-6709.2011.01214.x](https://doi.org/10.1111/j.1551-6709.2011.01214.x) — pattern separation/completion 与巩固。
- Teyler, T. J. & Rudy, J. W. (2007). The hippocampal indexing theory and episodic memory. *Hippocampus* 17(12):1158-1169. [DOI:10.1002/hipo.20350](https://doi.org/10.1002/hipo.20350) · [PubMed 17696170](https://pubmed.ncbi.nlm.nih.gov/17696170/) — CA3 pattern completion（relevance 维基底）。
- Howard, M. W. & Eichenbaum, H. (2013). The hippocampus, time, and memory across scales. *J. Exp. Psychol. Gen.* 142(4):1211-1230. [DOI:10.1037/a0033621](https://doi.org/10.1037/a0033621) · [PMC3982793](https://pmc.ncbi.nlm.nih.gov/articles/PMC3982793/) — 时间细胞/时间背景漂移（recency 维基底，近幂律）。
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annu. Rev. Neurosci.* 27:1-28. [DOI:10.1146/annurev.neuro.27.070203.144157](https://doi.org/10.1146/annurev.neuro.27.070203.144157) · [PubMed 15217324](https://pubmed.ncbi.nlm.nih.gov/15217324/) — 杏仁核/NE 调制巩固与检索（salience 维基底）。
- Hardt, O., Nader, K. & Nadel, L. (2013). Decay happens: the role of active forgetting in memory. *Trends Cogn. Sci.* 17(3):111-120. [DOI:10.1016/j.tics.2013.01.001](https://doi.org/10.1016/j.tics.2013.01.001) · [PubMed 23369831](https://pubmed.ncbi.nlm.nih.gov/23369831/) — 主动遗忘衰退（不物删的神经支持）。
- Ranganath, C. & Ritchey, M. (2012). Two cortical systems for memory-guided behaviour. *Nat. Rev. Neurosci.* 13(10):713-726. [DOI:10.1038/nrn3338](https://doi.org/10.1038/nrn3338) · [PubMed 22992647](https://pubmed.ncbi.nlm.nih.gov/22992647/) — 前颞叶(情绪显著)/后内侧(情景背景)双系统（salience/relevance 分工）。

**心理学席**
- Murdock, B. B. (1962). The serial position effect of free recall. *J. Exp. Psychol.* 64(5):482-488. [DOI:10.1037/H0045106](https://doi.org/10.1037/H0045106) — U 形系列位置效应（primacy 失真依据）。
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychol. Rev.* 63(2):81-97. [DOI:10.1037/h0043158](https://doi.org/10.1037/h0043158) — chunk 容量。
- Cowan, N. (2001). The magical number 4 in short-term memory. *Behav. Brain Sci.* 24(1):87-114. [DOI:10.1017/S0140525X01003922](https://doi.org/10.1017/S0140525X01003922) — 注意焦点≈4 chunks。
- Cowan, N. (2010). The magical mystery four. *Curr. Dir. Psychol. Sci.* 19(1):51-57. [DOI:10.1177/0963721409359277](https://doi.org/10.1177/0963721409359277) · [PMC2864034](https://pmc.ncbi.nlm.nih.gov/articles/PMC2864034/) — 3-5 unit 中央存储上限。
- Baddeley, A. (2000). The episodic buffer: a new component of working memory? *Trends Cogn. Sci.* 4(11):417-423. [DOI:10.1016/S1364-6613(00)01538-2](https://doi.org/10.1016/S1364-6613%2800%2901538-2) · [PubMed 11058819](https://pubmed.ncbi.nlm.nih.gov/11058819/) — 情景缓冲区有限容量。
- Tulving, E. & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. *Psychol. Rev.* 80(5):352-373. [DOI:10.1037/h0020071](https://doi.org/10.1037/h0020071) — 编码特异性（情绪线索检索依据）。
- Tulving, E. (2002). Episodic memory: from mind to brain. *Annu. Rev. Psychol.* 53:1-25. [DOI:10.1146/annurev.psych.53.100901.135114](https://doi.org/10.1146/annurev.psych.53.100901.135114) — 情景/语义划分。
- Murre, J. M. J. & Dros, J. (2015). Replication and analysis of Ebbinghaus' forgetting curve. *PLOS ONE* 10(7):e0120644. [DOI:10.1371/journal.pone.0120644](https://doi.org/10.1371/journal.pone.0120644) — 遗忘曲线现代重复（幂律方向）。
- Anderson, M. C., Bjork, R. A. & Bjork, E. L. (1994). Remembering can cause forgetting. *J. Exp. Psychol. LMC* 20(5):1063-1087. [Bjork Lab PDF](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Anderson_RBjork_EBjork_1994.pdf) — 提取诱发遗忘。

**数学席**
- Anderson, J. R. (1983). A spreading activation theory of memory. *J. Verbal Learn. Verbal Behav.* 22(3):261-295. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022537183902013) — ACT-R 激活方程（频率·新近·关联）。
- Wixted, J. T. & Ebbesen, E. B. (1991). On the form of forgetting. *Psychol. Sci.* 2(6):409-415. [DOI:10.1111/j.1467-9280.1991.tb00175.x](https://doi.org/10.1111/j.1467-9280.1991.tb00175.x) — 幂律忠实/指数失真。
- Wixted, J. T. & Ebbesen, E. B. (1997). Genuine power curves in forgetting. *Mem. Cognit.* 25(6):731-739. [Springer](https://link.springer.com/article/10.3758/BF03211316) · [PubMed 9337591](https://pubmed.ncbi.nlm.nih.gov/9337591/) — 个体层排除幂律假象。
- Parr, T. & Friston, K. J. (2017). Working memory, attention, and salience in active inference. *Sci. Rep.* 7:14678. [DOI:10.1038/s41598-017-15249-0](https://doi.org/10.1038/s41598-017-15249-0) · [PubMed 29116142](https://pubmed.ncbi.nlm.nih.gov/29116142/) — salience=认识论可供性，与 attention(精度)区分（写入门控=精度加权 PE 依据）。
- Mihaylova, L. et al. (2017). Distributed multisensor data fusion under unknown correlation and data inconsistency. *Sensors* 17(11):2472. [DOI:10.3390/s17112472](https://doi.org/10.3390/s17112472) · [PMC5713506](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5713506/) — 共享信源未建模→联合精度高估（double counting 依据）。

**计算机科学席**
- Packer, C. et al. (2023). MemGPT: towards LLMs as operating systems. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) — LLM-in-loop page-in/out（本项目刻意不选的对照）。
- Rasmussen, P. et al. (2025). Zep: a temporal knowledge graph architecture for agent memory. [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) — RRF/MMR 确定性重排 + 时序失效（解法 B 算法参考）。
- Carbonell, J. & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98*. [DOI:10.1145/290941.291025](https://dl.acm.org/doi/10.1145/290941.291025) — MMR 原始公式（diversity 维度）。

> 项目规则锚：`.claude/rules/memory-rules.md`(#1 节流 / #2 显式 scope / #3 运行态分离 / #5 封装边界) · `.claude/rules/orchestration-rules.md`(state 不放大对象) · `CLAUDE.md`(确定性热路径·记忆纪律) · `ai-docs/pitfalls.md`(语义是可选侧信道·读节点不写) · 记忆 `analysis-results-first-no-intervene`。
