# 对话情景记忆设计决策（科学家议会综合 · 2026-06-25）

- 日期：2026-06-25
- 议题：把"对话经历写进语义记忆 + 召回回灌对话"做扎实——写什么 / 何时 / 什么 scope / 召回怎么用，既让仿生人"记得过去交流"又守 memory-rules 节流红线。
- 评审席位：心理（情景/自传记忆）· 神经（海马编码/巩固）· CS（红线/可行性，必到）；主持席综合。上次议会明确把"写 transcript/摘要进语义记忆"列为**必回议会再落**，本轮即评。
- 用户决定（不在评审范围）：**A** = `--chat` 默认开 `sqlite_vec` 语义后端；**B** = 把对话经历做成有用可召回记忆。main.py `_run_chat` 是用户直接验证入口、要开箱即用。
- 治理：议会只读、强制引文、不改代码、不介入数据产生。本文件是唯一写动作。

## 各席判定汇总

| 席位 | 判定 | 关键引用 |
| --- | --- | --- |
| 心理（情景/自传记忆） | **简化**（episode 只写情感坐标+刺激名→"记得聊过什么"几乎无用） | Tulving 2002 · Brainerd&Reyna 2002 · Neisser 1981 · Conway 2005 · arXiv 2506.08184 |
| 神经（海马编码/巩固） | **简化**（每轮无差别写、无显著性门控；召回缺 mood-congruent 偏置） | McGaugh 2004 · Lisman&Grace 2005 · Grande 2019 · Faul&LaBar 2022 |
| CS（红线/可行性·必到） | **WARN×3**（A 开关漏设 / B 失败隔离缺失 / dedup 缺失），红线未触 | memory-rules #1 · pitfalls |

**无实质跨学科冲突**：心理（写 gist/聊什么+结论）与神经（显著性/情绪加权门控）互补、共同指向"高情绪显著性的对话 gist 才值得持久化"；召回三席一致改"全量倒"为"选择性+情绪偏置+阈值"。唯一需裁定点：gist 内容怎么生成才不触 LLM/meta 红线（见下）。

## 决策

### B-1 episode 写什么：确定性三层拼接（守红线）
现状 `supervisor.py:49-54` 只写"用户对刺激X表现Y情绪(v/a/value)"——缺对话 gist。改为三层**确定性字段拼接**：
1. **对话 gist**：`state.stimulus.text`（用户原话，截 200 字符；无则退 `stimulus.name[:40]`）+ `state.language_text`（助手本轮回应，截 200）——"你说：… / 我说：…"。
2. **情绪坐标**：`text_label(v,a)` + (valence, arousal)。
3. **显著性元数据**：`affect_precision` + `ignited_streams` + `value_estimate`。

形如：`你说：<user_text[:200]> / 我说：<language_text[:200]> | 情绪={label}(v,a) | precision=… | streams=… | value=…`

### B-2 红线裁定：gist 用确定性拼接、**不调 LLM 生成摘要**
- `stimulus.text`/`name`=用户原始输入（观测量）；`language_text`=LanguageAgent 已产出的下游产物（观测，非介入生成）；`text_label`=确定性查表。整个拼接无 LLM 调用、无 meta 介入——等价于写结构化日志，不产生新情感/语言数据，**守红线**（CLAUDE.md「禁止 LLM/meta 介入运行时数据产生」、记忆 `analysis-results-first-no-intervene`）。
- **禁止**：另开 LLM 调用"生成对话摘要"——那会让 LLM 介入记忆数据产生，违红线。

### B-3 何时写：显著性门控（神经忠实 + 降成本/抖动）
海马编码受情绪唤起(McGaugh)+新颖性(Lisman&Grace)门控、平淡淡化。现状每轮无差别写。改为：写 episode 前算 `salience = (affect_precision or 0.0) × (abs(rpe) if rpe is not None else 0.5)`，`salience < SALIENCE_THRESHOLD`（默认 0.15、env 可配）则**跳过 write_episode**，仅写结构化 `write`（标量事实不门控、保全量）。门控只用 AffectState 已有确定性信号（precision/rpe/ignited_streams），无新模块、无 LLM。

### B-4 scope：保持 `Scope.USER` + `key=user_id`（=thread，防串味）
三席确认合理（跨会话长期记忆 + thread 隔离）。

### B-5 召回：选择性 + 情绪偏置 + 阈值（非全量倒）
现状 `memory_recall.py:53-56` 用 `stimulus.name` 作线索、`recall(limit=5)` 全量。改：
- **查询线索**拼当前情绪标签（mood-congruent，Faul&LaBar 2022）：`query=f"{stimulus.name} {text_label(mood[0],mood[1]) if mood else ''}"`（mood 关时退化）。
- **相关性阈值**：`sim_threshold`（默认 0.65、env 可配）过滤低相关，防 proactive interference（arXiv 2506.08184）。
- limit 保持 5，配阈值后实际返回自然缩减。

### A 工程必修（两条·BLOCK 级，不修则 A 等于没做/会崩）
- **必修①**：`main.py _run_chat` 补 `os.environ.setdefault("ZERO_SEMANTIC_BACKEND", "sqlite_vec")`（与现有两行 setdefault 并排）。现状缺它→`build_semantic_store()` 返 None→`recall_enabled=True` 形同虚设、B 从未生效。（缺 openai 时 `_sqlite_vector_store` 告警回退 None、零回归不崩。）
- **必修②**：`SqliteVectorStore._embed` 及 `MemoryClient.write_episode`/`recall` 加 `try/except Exception → logger.warning` + 空结果/跳过、**不向上抛**。现状 embedding API 失败会冒泡崩 supervisor 节点→崩主对话，违"语义是可选侧信道绝不拖垮主管线"。

### dedup（可选优化·建议同批次落）
`SqliteVectorStore.add_episode` 写入前检索 top-1 相似 episode，余弦 > `DEDUP_THRESHOLD`（默认 0.92）则跳过，防长对话近义 episode 堆积降召回质量。轻量确定性、无 LLM。

## 可接受的刻意简化（及代价）
- gist 用确定性拼接而非 LLM 摘要（守红线 > 语义精炼）：拼接有冗余、embedding 质量略低于真摘要。
- 用 sqlite_vec 而非 Graphiti 图谱抽取（小体量轻量优先）：无实体/关系跨 episode 关联。
- salience 门控用 `precision×|rpe|` 简化公式：边界粗，高 precision+低 rpe 可能漏写。
- **无完整巩固/遗忘曲线**（McGaugh 睡眠巩固 / Ebbinghaus 离线衰减批处理）：**明确不在本轮**，列路线图后期；代价=久远 episode 不衰减、持续占召回空间。

## 悬而未决（工程落地时定）
1. `SALIENCE_THRESHOLD`/`sim_threshold`/`DEDUP_THRESHOLD` 默认值（建议 0.15/0.65/0.92）需实跑校准，先取保守值、走 env。
2. `language_enabled=False`（词典回退）时 `language_text` 是模板串、噪声大——是否此时 episode 只写用户 gist、跳过 language_text。
3. `mood_enabled=False`（--chat 现状）时 mood=None，mood-congruent 召回退化——是否改用 `emotion`(emotion_decay 输出)作情绪线索（该信号在 _run_chat 图外、需显式传入）。

## 结论：NEEDS-CHANGES（两条必修修完即进 /engineer）
必修①②修完后，B 的全部设计（确定性三层 episode + 显著性门控 + 选择性召回 + dedup）作同批次 /engineer 任务落地。完整巩固/遗忘明确不在本轮。

## 引文（各席现场核验，带链接）
**心理席**
- Tulving, E. (2002). Episodic memory: From mind to brain. *Annu. Rev. Psychol.* 53:1-25. [DOI:10.1146/annurev.psych.53.100901.135114](https://doi.org/10.1146/annurev.psych.53.100901.135114)
- Brainerd, C. J. & Reyna, V. F. (2002). Fuzzy-trace theory and false memory. *Curr. Dir. Psychol. Sci.* 11(5):164-169. [DOI:10.1111/1467-8721.00192](https://doi.org/10.1111/1467-8721.00192)
- Neisser, U. (1981). John Dean's memory: A case study. *Cognition* 9(1):1-22. [DOI:10.1016/0010-0277(81)90011-1](https://doi.org/10.1016/0010-0277%2881%2990011-1)
- Conway, M. A. (2005). Memory and the self. *J. Mem. Lang.* 53(4):594-628. [DOI:10.1016/j.jml.2005.08.005](https://doi.org/10.1016/j.jml.2005.08.005)
- Kensinger, E. A. (2009). Emotion and autobiographical memory（情绪增强记忆综述）. [PMC2852439](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2852439/)
- Proactive interference / 过量召回干扰（AI 记忆）. [arXiv:2506.08184](https://arxiv.org/abs/2506.08184)

**神经席**
- McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annu. Rev. Neurosci.* 27:1-28. [PubMed 15217324](https://pubmed.ncbi.nlm.nih.gov/15217324/)
- Lisman, J. E. & Grace, A. A. (2005). The hippocampal-VTA loop. *Neuron* 46(5):703-713. [PubMed 15924857](https://pubmed.ncbi.nlm.nih.gov/15924857/)
- Grande, X. et al. (2019). Holistic recollection via pattern completion involves hippocampal subfield CA3. *J. Neurosci.* 39(41):8100-8111. [PMC6786823](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6786823/)
- Faul, L. & LaBar, K. S. (2022). Mood-congruent memory revisited. *Psychol. Rev.* [PMC10076454](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10076454/)
- Dolcos, F., LaBar, K. S. & Cabeza, R. (2006). The memory-enhancing effect of emotion: functional neuroimaging evidence. *Prog. Brain Res.* 156:135-149. [DOI:10.1016/S0079-6123(06)56007-8](https://doi.org/10.1016/S0079-6123%2806%2956007-8)

**CS 席**
- 项目规则 `.claude/rules/memory-rules.md #1`（任务完成节点节流）；`ai-docs/pitfalls.md`（记忆写入未节流刷爆图谱 / Graphiti 侧信道不进确定性热路径）。
- Tulving, E. (1972). Episodic and semantic memory. In *Organization of Memory*, Academic Press. [出版社条目](https://doi.org/10.1016/B978-0-12-301850-2.50001-2)

## 相关代码位置（只读）
- `src/orchestration/supervisor.py:49-54`（write_episode 内容 + 待加显著性门控）
- `src/orchestration/memory_recall.py:53-56`（recall 线索 + 待加情绪偏置/阈值）
- `src/storage/backends/semantic.py`（SqliteVectorStore `_embed`/`add_episode`/`search` + 待加 try/except、sim_threshold、dedup）
- `src/memory/client.py:65-111`（write_episode/recall + 待加 try/except 隔离）
- `main.py:186-187`（_run_chat 缺 ZERO_SEMANTIC_BACKEND setdefault）
