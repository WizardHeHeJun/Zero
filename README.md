# Zero — 情感引擎驱动的 AI 数字人

> 让机器**带着情绪**说话。Zero 以一套**情感引擎**为内核、以 **LLM** 为语言外壳：每一句话先被读成情绪、在引擎里按人类情感动力学演化，再由语言与表情把这份情绪自然地漏出来——不是让模型"扮演"情绪，而是让情绪真实地参与生成。

情感引擎融合了五个学科的建模视角：

- **数学** — 贝叶斯主动推断、动力系统、在线价值学习
- **心理学** — OCC 评价理论、效价-唤醒环状模型、情绪调节、评价性条件作用
- **生物学** — 面部动作单元（FACS）、自主神经生理反应
- **神经科学** — 预测编码、全局工作空间点燃、显著网络门控、杏仁核多通路、多巴胺奖赏预测误差
- **计算科学** — 多 Agent 编排（LangGraph）、多网络并行

---

## 情感引擎是怎么做的

把"人产生并表达情绪"的过程，建模成一条**贝叶斯流水线**——感知一句话、推断出此刻的情绪、让它随时间演化、再分两路外化为语言和表情。

![情绪引擎框架图](docs/v2/framework-current.png)

### 1. 评价桥：把话读成情绪

每一句输入先经**评价桥**反推出效价-唤醒坐标 `(v, a)`——一句夸奖是正效价、一句挑衅是负效价高唤醒——作为刺激喂给引擎。

### 2. 情感引擎核心：贝叶斯主动推断 + 显著度门控工作空间

引擎不是简单地"查表给情绪"，而是把多条**并行的功能流**竞争整合成一个全局情绪状态：

- **OCC 评价流** — 按目标契合度 / 标准符合度 / 对象吸引力给出情绪先验；
- **价值流** — 在线的 TD 奖赏预测误差与精度（对惊喜、对确定性的敏感）；
- **生存流** — 快速、低精度的亚符号信号（突如其来的巨响先于"这是什么"就拉高唤醒）；
- **语义流** — 文本/语义读出的情绪作为一条**独立、低精度**的高阶先验汇入（语言是高阶皮层的 top-down 预测，与评价、感官流并列竞争而非互相覆盖）；
- **显著度门控点燃** — 各流各出一个 `(均值, 精度)`，由显著网络打分，只有**过阈的流被"点燃"广播**进全局工作空间（全局工作空间理论的 ignition），其余停留局部、不空播；
- **精度加权融合 + 后验采样** — 点燃的流按精度加权融合，采样出此刻的**瞬时情绪 e\***（随机性让同一刺激也有细微波动）；读出也可切**稳定模式**（取后验均值而非单次采样），让情绪只跟刺激走、不被单样本噪声带得逐轮乱跳（见配置 `ZERO_AFFECT_READOUT`）。

### 3. 三时间尺度：情绪会退、态度会沉淀

人的情绪不是一锤子，而是**多个时间尺度叠加**：

- **瞬时 `e*`** — 每个刺激当下采样出的情绪；
- **快变 `emotion`** — 短时情绪，被 e\* 冲击后**几轮内向基线衰退**（怒火飙起后会回落；衰退太慢反而是病理性的情绪惯性）。对外表达取的是它；
- **慢变 `attitude`** — 对**特定对象**的长期态度，按情绪缓慢累积、多轮才成形，是快变情绪衰退回归的基线。**持续**被冒犯才会真的变冷，偶尔被呛一下会过去。**只有态度被持久化**，重启后情绪归于态度基线。
- **稳态回弹** — 情绪与态度都带一份**回到平静的拉力**（向个体中性基线弱回归）：再热烈或再低落，只要没有持续刺激就会慢慢回稳，不会"越聊越上头"或陷在某个极端里出不来（affective homeostasis；情绪基线本身也是态度与中性的混合，不随态度无限上漂）。
- **唤醒双向 · 习惯化 · 分寸** — 唤醒（arousal）也是**双极**的：平淡对话会主动**降到静息**（不只是不涨）、重复互动会**习惯化**（新鲜感递减）、对刚认识的人有**分寸感**（不因聊久了就无端亲密）——从根上防「与内容无关地越聊越暧昧」（科学家议会 seeking 吸引盆两轮裁决；默认关、按需开旋钮，见配置全表）。

于是对话有了"脾气"：被骂会不快、道歉能缓和、但一时的情绪不会永久定义这段关系，也不会因一路投入就单调滑向极端。

### 4. 双路语言：命题靠 LLM，情绪靠"漏"

借鉴语言的皮层/皮层下双通路：

- **Pull（皮层 · 随意）** — LLM 负责**命题内容**，根据上下文与检索把"要说什么"组织成话；
- **Push（皮层下 · 不随意）** — 情绪经**用词倾向 / logit 偏置 / 隐状态 steering** 自动**漏进**输出，而不是给模型一句"请表现得很生气"。

情感**辅佐而非替换**语言：它改变措辞的温度、节奏、边界感，让回应自然带情绪，而非戏剧化地演情绪。

### 5. 表达双通路：自发与随意 × 多通道

最终表现分**自发**（真情流露）与**随意**（社交掩饰）两条通路，落到多个表达通道——面部动作单元（FACS AU）、文本标签、生理信号、语音韵律。

### 6. 记忆 / 持久：短时注意力 ⊗ 长时记忆

多层记忆让数字人**跨重启记得你**，并在"当下注意得过来"与"长期记得住"之间架一座桥：

- **对话运行态** — transcript 与对此人的长期态度落本地 SQLite，重启续上；态度还作为先验**偏置当下的情绪评价**（持续被冒犯，连初见的反应都会变冷）。
- **短时注意力（工作记忆窗）** — 喂给 LLM 的上下文不是简单截最近 N 轮，而是**首因 + 近因的 U 形窗**：既记得"第一次见面说的话"、也记得最近几轮（借鉴系列位置效应，避免单调截断丢掉开场）。
- **情景记忆 + 三维召回** — 把对话经历按**情绪显著性**择要写成情景 episode（平淡的不记，借鉴海马"情绪/新颖性门控"）；召回时按 **新近性 × 相关性 × 重要性** 三维加权排序（幂律时序衰减 · 语义相似 · 写入显著度），高分的旧记忆**升入注意力预算、与近期对话同台竞争**（对应皮层记忆重激活），而非旁路堆砌——于是它**记得你聊过什么**、且只在相关时想起来。
- **约定记得住、记不清就直说** — 含时间 / 地点 / 承诺的内容，即使当时情绪平淡，也经**语义重要性通道**单独入库（不被情绪显著性门挡掉），日后能答"我们约的几点"；而真记不清时会**坦诚说"不记得"**，不编造、也不拿脾气搪塞（事实优先于情绪）。
- **遗忘是特性** — 长期事实带**时序失效**（新事实使旧失效）、情景库有**容量上限**，靠自然沉降而非物理删除来遗忘；确定性图谱 + 语义召回侧信道并存，语义侧信道失败绝不拖垮主对话。

> 记忆写什么/何时写/怎么召回/怎么排序，都经**跨学科科学家议会**评审定调（强制现场引文），且**全程确定性、不让 LLM 替数字人"编造"或"挑选"记忆**。一键恢复出厂：`python -m tools.reset_db --yes`。

![记忆架构：注意力↔记忆桥（显著性写 · 三维召回 · 注入预算 · 时序遗忘）](docs/v2/memory-architecture.png)

### 7. 指定人格：不必从零认识一个人

**性格该预置，关系才靠相处长**——可以给数字人指定一份**人格**，免去每次从一张白纸开始：

- **人设卡** — 名字 / 背景 / 口吻 / 与你的关系，作为身份注入对话（"它是谁"）；
- **气质底色** — 习惯性情绪基线、反应快慢与情绪恢复速度（偏暖还是偏冷、易激动还是沉稳），是性格的"生理底色"，落到引擎的态度 / 情绪参数上；
- **预置关系** — 初见即已有的态度（一开始就熟络 / 在意某人）+ 预灌的共同记忆（"我们一起去过海边"），跳过从零相处。

不指定时即**中性无偏人格**，行为与从前逐字一致。其中"什么性格对应怎样的情绪参数"（大五人格 → 情感维度的具体映射）属研究决策，交由科学家议会定调——引擎只提供可调旋钮，不替算法臆断。

![人格注入：Persona 三层各接到哪（人设卡→语言 / 气质→引擎 / 预置关系→记忆）](docs/v2/persona-injection.png)

---

## 项目运作流程：LLM ⊗ 情感引擎

把上面的情感引擎接进一次完整对话——**LLM 只在「输入」「输出」两端与它结合**（图中两个蓝框）：输入端把你的话**读成情绪**，输出端**被情绪调制着说话**；夹在中间产生情绪的是那套**确定性引擎**（红框，无 LLM），LLM 既不进情绪计算热路径、也不替数字人"编造"记忆。

![项目运作流程图](docs/v2/runtime-flow.png)

### 关键接口各自的作用（LLM ⊗ 情感接点已标注）

| 接口 / 节点 | 作用 |
| --- | --- |
| `ConversationModel.appraise_text(text) → (v,a)` | **评价桥 · LLM 输入接点**：把你的话读成情绪坐标，作为刺激喂给引擎 |
| `ConversationModel.converse(history, affect, *, retrieved, push)` | **自然对话 · LLM 输出接点**：按当前情绪 + 召回背景生成回应，情绪经用词倾向自然漏进措辞 |
| `LanguageModel.generate → LanguageDraft` | 图内 `language` 节点协议：研究模式的 affect↔language 双向收敛回路（`python main.py --llm`） |
| `ChannelDecoder`（鸭子类型注入） | 表达通道解码器：`(v,a)` → 韵律 / 生理 / 表情，可换成训练好的网络，编排层不依赖 torch |
| `MemoryClient`：`write_episode` / `recall` · `write` / `query` | 记忆读写 API：语义情景记忆（显著性写入 / 选择性召回）+ 确定性长期倾向（图谱·时序失效）；**上层不直连图谱**、写入只在任务完成节点（节流） |
| 图节点链 `perception → appraisal → value → affect_core → mood → expression → supervisor`（+ `memory_recall`） | 情感引擎各环：感知 → 评价先验(含长期态度/召回偏置) → 价值学习 → 显著度门控融合采样 `e*` → 慢心境 → 多通道表达 → 任务完成节流写记忆 |
| 入口 `build_graph` · `runner.ConversationSession` · `main.py` | 装配并编译图 · 多轮会话基元（mood/价值/记忆跨轮持久）· CLI（`python main.py` 即进对话） |

> 各接口均**协议化、可注入**（真 LLM / 占位模板 / steering 后端、真网络解码器、记忆后端都按协议替换），编排层不绑定具体 SDK——这是"先把对话做扎实、再逐步接多模态"而不动内核契约的底座。

---

## 预留给未来的通道

现阶段以**文本输入、情绪化文本输出**跑通整条回路，同时把若干扩展点的**接口先留好**，未来逐步接入而不动内核契约：

| 方向 | 现在 | 预留的未来通道 |
| --- | --- | --- |
| **表达解码器** | 各通道用确定性占位函数 | 每个通道可**换成可训练网络**（韵律 RAVDESS / 生理 WESAD / 表情 AffectNet / 文本→VAD EmoBank），经鸭子类型协议注入，编排层不依赖 torch |
| **输入感知** | 文本 → `(v, a)` | 视觉图像 / 心电（ECG）/ 语气 / 面部表情 → 更丰富的多通道感知 |
| **输出形态** | 情绪化文本 + 通道值 | Live2D 形象 / 情感 TTS 等多模态外化 |
| **记忆与经历** | 对话+态度落盘、情景择要落库 + 新近×相关×重要三维召回 + 首因/近因注意力窗 | 完整睡眠巩固/遗忘曲线、稳定人格/自我模型、跨会话人物画像 |
| **运行后端** | 默认本地（内存 / SQLite） | env 一键切容器化 Postgres / Neo4j，接入真实图谱与运行态持久 |

---

## 项目结构

三层架构，依赖**单向**：编排 → 记忆 → 存储。

```text
Zero/
├── main.py                  # CLI 入口：默认 python main.py 即进对话；--workspace / --llm / --trace
├── src/
│   ├── orchestration/       # 编排层：StateGraph 装配 + 运行入口
│   │   ├── graph.py         #   build_graph：节点装配 + 条件边路由
│   │   ├── state.py         #   AffectState / Stimulus（结构化 state，含 recalled_facts）
│   │   ├── supervisor.py    #   协调 + 任务完成节流写记忆 + first_contact 首因标记
│   │   ├── memory_recall.py #   长期倾向回灌先验 + 召回三维重排（新近×相关×重要，Hill 归一）
│   │   ├── chat_driver.py   #   交互对话核心：两时间尺度情绪 + U形注意力窗 + 高显著召回注入
│   │   └── runner.py        #   跑刺激序列 + 多轮对话会话（ConversationSession）
│   ├── agents/              # 各 Worker（节点契约 (state) -> dict 只回增量）
│   │   ├── affect_math.py   #   数学内核：OCC/TD/精度/高斯融合·工作空间·三时间尺度
│   │   ├── perception.py · appraisal.py · value.py
│   │   ├── affect_core.py   #   主动推断·后验采样 e*（并行流竞争 + ignition）
│   │   ├── mood.py          #   慢变心境双稳动力学
│   │   ├── regulation.py · expression.py   # 掩饰 + 双通路·多通道输出
│   │   ├── language.py · language_openai.py   # 语言生成+双向回路 / ConversationModel 协议 / 评价桥 / 自然对话
│   │   ├── persona.py       #   指定人格：人设卡(L1)+气质底色(L2)+预置关系(L3)，默认中性零回归
│   │   ├── emotion_lexicon.py    #   细粒度情绪词 / 动机系统 / VAD 词典桥 / 时间包络
│   │   ├── language_steering.py  #   VA steering 适配器（开放权重）
│   │   ├── models/          #   可训练 torch 解码器（expression/prosody/physiology/facs/text）
│   │   └── datasets/        #   DataLoader：synthetic / ravdess / wesad / emobank / facs
│   ├── memory/              # 记忆层：读写 API（显式 scope、任务完成节流、后端失败隔离 + Fact.sim）
│   ├── storage/             # 存储层（最底层）：运行态 + 长期记忆，env 选后端
│   │   ├── checkpointer.py  #   memory / sqlite(异步 AsyncSqliteSaver) / postgres(待异步接线)
│   │   ├── graph_store.py   #   门面 + 工厂
│   │   ├── conversation_log.py  #   --chat 对话运行态：transcript + 跨重启 attitude 落本地 SQLite
│   │   └── backends/        #   deterministic（InMemory/Sqlite/Neo4j）+ semantic（Graphiti/SqliteVector）
│   └── observability/       # 横切：统一日志 setup_logging（每启动落 logs/、级别可配、入口无关）
├── tests/                   # 单测 + 行为/记忆回归
├── scripts/                 # 训练脚本 train_*.py + 端到端 demo_pipeline.py + 验证 verify_*.py（如 verify_affect_readout 实测 map 读出消翻号）
├── tools/                   # 运维脚本（reset_db.py 清库）
├── docs/                    # 对外框架图（v1/v2 谱系 + 运作流程图，详见 docs/README.md）
├── notes/                   # 研究笔记 / 科学家议会决策 / 工程实践（情感数学·文本输出·工作空间·路线图·记忆路由…）
├── .env.example                                     # 配置模板（cp 为 .env 启用）
├── personas/                                        # --chat 人格卡目录：*.example.json 模板随仓库共享 / 个人 *.json 走 gitignore；放多份 persona 改 ZERO_PERSONA_FILE 即切换
├── Dockerfile · docker-compose.yml                  # 容器化部署
└── pyproject.toml · environment.yml                 # 依赖与环境（core + ml/llm/nlp/steer/db 默认装；graphiti 按需）
```

---

## 快速开始

环境用 conda 管理（环境名 `affective-expression`，Python 3.12，依赖口径以 `pyproject.toml` 为准；也支持 `uv sync`）。

```powershell
# 1. 建环境
conda env create -f environment.yml
conda activate affective-expression

# 2. 直接对话：情感引擎 ⊗ LLM（缺 LLM key 自动回退词典 + 模板，仍演示情绪演化）
python main.py

# 3. 看显著度门控工作空间：每个刺激点燃了哪些并行流
python main.py --workspace

# 4. 核心管线 (v,a) 轨迹 JSON
python main.py --trace
```

> **对话时每轮会打印一行 trace**：`你这句≈(v,a)`（你这句被读出的情绪坐标）｜`情绪=<词>(v,a)`（数字人此刻的情绪）｜`对你的态度=(v,a)`（它对你的长期态度）——一眼看清引擎在想什么。

接**真 LLM**（OpenAI 兼容接口，本地 vLLM / 第三方网关皆可）需 `llm` extra 并在 `.env` 配置——配置只走 `.env`，代码不写死模型默认：

```powershell
pip install -e ".[llm]"
# .env 内：
#   ZERO_OPENAI_API_KEY=sk-...                  # 必填
#   ZERO_OPENAI_MODEL=<你的 key 可访问的模型 id>  # 必填
#   ZERO_OPENAI_BASE_URL=https://.../v1          # 可选
python main.py            # 真模型对话
python main.py --llm      # 四情绪场景的文本输出情绪验证（批处理）
```

**真网络化**（把表达通道换成训练好的网络）需 `ml` extra，数据集获取见 **[DATASETS.md](DATASETS.md)**：

```powershell
pip install -e ".[ml]"
python -m scripts.train_prosody --root data/ravdess --epochs 300   # 权重存 artifacts/，再注入管线
python -m scripts.demo_pipeline                                    # 端到端：合成训练 → 注入 → 跑（无需外部数据）
```

> **不想自己训练？直接用现成权重**：真实数据训练好的权重已随 Release 提供，拿来即用——从 [`weights-v0.1`](https://github.com/WizardHeHeJun/Zero/releases/tag/weights-v0.1)（稳定版 [`v0.1.0`](https://github.com/WizardHeHeJun/Zero/releases/tag/v0.1.0) 附件是同一份）下载 5 个 `.pt` 放入仓库根目录 `artifacts/`（已 gitignore），各 `load_*` / `scripts/*`（如 `demo_pipeline`）自动加载；缺某通道回退内置默认 / 占位、不影响其它。
> - 五通道：`text_affect_regressor.pt` / `text_affect_regressor_st.pt`（文本→(v,a)，词袋 / 句向量，EmoBank）· `prosody_decoder.pt`（(v,a)→韵律，RAVDESS）· `physiology_decoder.pt`（(v,a)→生理，WESAD）· `expression_decoder.pt`（(v,a)→表情 FACS，demo）。

> **日志与排障**：每次启动落一份 `logs/zero-<时间戳>.log`；排障时 `ZERO_LOG_LEVEL=DEBUG python main.py ...` 可看每轮引擎 `e*`、记忆读写、LLM 请求/响应等详情，默认 `INFO` 保持安静、不打扰对话。对话另落一份**人读日志** `logs/conversation-<时间戳>.log`（每轮 user/Zero 原文 + 评价/情绪/态度 trace，默认开、`ZERO_CONVERSATION_LOG=0` 关且不落任何对话内容）。

> **开发/测试**：`pytest`（全套回归）· `ruff check . && ruff format .`（风格）· `mypy src`（类型）——保存时基础检查自动跑。

---

## 配置（`.env`）

所有运行配置都走 `.env`（复制 [.env.example](.env.example) 起步），代码不写死模型/后端默认。**不设任何变量即全内存占位、零依赖可跑**；`.env.example` 里每个变量都有一行速记，下面按用途分组给出完整说明。

**怎么读 `.env`（三类）**：

- **【必填】** 只有 `ZERO_OPENAI_API_KEY` + `ZERO_OPENAI_MODEL`（接真 LLM 用；缺了 `--chat` 自动回退词典+模板，仍能跑）。
- **后端选择**（顶部各组）：`.env.example` 里给出的赋值就是**各自默认值**，写不写效果一样，想切落盘/真库才改。
- **可选旋钮**（底部）：默认**注释掉** = 用内置默认；取消注释才覆盖。其中 ⭐ 是**数字人推荐开**——`ZERO_PERSONA_FILE`（治"上来就编造关系"）+ `ZERO_AFFECT_READOUT=map`（治情绪标签逐轮翻号），两个足矣；`ZERO_APPRAISE_CALIBRATE` 视模型可选（强模型如 deepseek 本就把敌意读得够负、可不开）。

### 运行后端

运行态 Checkpointer 与 长期记忆图谱**各自独立选后端**，可任意组合；默认都在内存，落盘 / 真后端按需开。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZERO_CHECKPOINT_BACKEND` | `memory` | 运行态后端：`memory` / `sqlite` / `postgres` |
| `ZERO_CHECKPOINT_DB` | `data/checkpoints.sqlite3` | sqlite 后端的库文件路径 |
| `ZERO_PG_DSN` | — | postgres 后端 DSN |
| `ZERO_MEMORY_BACKEND` | `memory` | 长期记忆图谱（确定性 `(scope,key)` 失效）：`memory` / `sqlite`（落盘）/ `neo4j`（需 `db` extra） |
| `ZERO_GRAPH_DB` | `data/graph.sqlite3` | sqlite 图谱的库文件路径 |
| `ZERO_NEO4J_URI` · `_USER` · `_PASSWORD` | `bolt://localhost:7687` · `neo4j` · `password` | neo4j 连接（`ZERO_MEMORY_BACKEND=neo4j` 或 Graphiti 用 neo4j 图库时生效） |

### LLM 接入（OpenAI 兼容）

语言层（评价桥 + 自然对话）与语义记忆 embedding **共用一处** OpenAI 兼容接口，最小配置见上方「快速开始」。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZERO_OPENAI_API_KEY` | — | 必填 |
| `ZERO_OPENAI_MODEL` | `qwen-flash` | **必填**，须是 key 有权限的真实模型 id（`limited` 等权限标签不是模型名、会 400；可用 `/v1/models` 列出，如 `qwen-flash` / `deepseek-v4-flash` / `gpt-5.5`） |
| `ZERO_OPENAI_BASE_URL` | `https://api.openai.com/v1` | 可指向 OpenAI / 本地 vLLM / Ollama / 第三方网关，留空用 SDK 默认 |

### 语义记忆侧信道（默认关）

确定性图谱之外，可叠一条**语义召回**侧信道（向量相似召回）；默认关，**侧信道失败绝不拖垮主对话**。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ZERO_SEMANTIC_BACKEND` | 关 | `sqlite_vec`（轻量、推荐）/ `graphiti`（深度集成，需 `graphiti` extra） |
| `ZERO_GRAPHITI_DB` | `neo4j` | Graphiti 图库，复用上面 `NEO4J_*` |
| `ZERO_GRAPHITI_MODEL` | `deepseek-v4-pro` | Graphiti 抽取实体 / 关系入图谱用的对话/推理 LLM |
| `ZERO_GRAPHITI_EMBED_MODEL` | `gemini-embedding-001` | 向量嵌入模型（`sqlite_vec` / `graphiti` 都用它做相似召回，须是 embedding 模型） |

> **文本情感回归**（输入侧，默认走词典 / LLM 评价桥）：置 `ZERO_TEXT_AFFECT_BACKEND=st` 改用训练好的回归头，须同时给 `ZERO_TEXT_AFFECT_MODEL_PATH`（如 `artifacts/text_affect_regressor_st.pt`）。

### 指定人格（`--chat`）

给数字人指定一份人格（能力详见上文「指定人格」一节）：`ZERO_PERSONA_FILE` 指向一个**人格 JSON**（默认不设 = 中性无偏人格、逐字现有行为）。仓库自带一份「诚实陌生人」模板 `personas/persona.example.json`（与真正的 `personas/persona.json` 同处一目录），`cp personas/persona.example.json personas/persona.json` 改改即用（想要多重人格就在 `personas/` 放多份、切换时改 `ZERO_PERSONA_FILE` 指向即可）。字段全可选（L1 人设卡 + L2 气质底色 + L3 预置关系）——**只想要人设卡就只写 `card` 一个字段**，不必写全、也不用往 `.env` 塞长文本：

```jsonc
{
  "name": "小津",
  "card": "你叫小津，是用户多年的老友……",   // L1 人设卡
  "setpoint": [0.1, -0.05],                  // L2 气质基线 (v,a)：略偏暖、偏平静
  "reactivity": 0.6,                         // L2 对刺激的即时反应增益（↑≈神经质）
  "recovery": 0.4,                           // L2 情绪恢复残留比例（↑=情绪退得慢）
  "initial_attitude": [0.3, 0.1],            // L3 首次接触的初始态度（已经喜欢这个人）
  "seed_memories": ["我们去年夏天一起去过青岛看海", "你不吃香菜"]  // L3 预灌的共同记忆
}
```

> L2 的「大五人格 → PAD 具体数值映射 / 预设人格库」属科学决策，须走 `/science-council` 设计门；本接口只提供旋钮 + 中性默认，不替算法拍板具体性格参数。

### 对话调优与排障（对症开旋钮）

对话不对劲时，多数能靠一两个旋钮解决——先按症状开，细节见下方全表：

| 症状 | 开什么 |
| --- | --- |
| 一上来就编造共同往事 / 假装认识你 | `ZERO_PERSONA_FILE`（给它身份 +「初次见面不编造」，见上文「指定人格」） |
| 情绪标签逐轮乱跳、与内容不符（敌意却标「兴奋」） | `ZERO_AFFECT_READOUT=map`（取后验均值、消采样翻号） |
| 敌意/负面被读得太轻 | `ZERO_APPRAISE_CALIBRATE=1`（**视模型**：强模型如 deepseek 本就够负、可不开） |
| 越聊越「上头」、情绪停在高位 | 调低 `ZERO_EMOTION_BASELINE_ATTITUDE_W`（加大回中性的拉力） |
| 越聊越「暧昧」/ 关系无端升温、与对话内容脱钩 | `ZERO_INTENSITY_FLOOR=0` + `ZERO_AROUSAL_BASELINE=-0.08` + `ZERO_ATTITUDE_REVERSION_A=0.4`（去 arousal 直流偏置，见下「越聊越暧昧」全表） |

> 两个**自查脚本**（无需改代码）：`python -m scripts.verify_affect_readout`（**无需 LLM**，实证 `map` 把翻号率从 ~20% 压到 0）；`python -m scripts.verify_appraise_calibration`（**需 LLM key**，按你的模型实测标定要不要开）。

### 微调旋钮·全表

默认开箱即用（仅 `HISTORY_*` / `EMOTION_BASELINE_ATTITUDE_W` 的默认改变 `--chat`，其余默认零回归 / 关）；设计依据见 [notes/](notes/) 议会纪要。

**① 数字人情绪 / 对话**

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `ZERO_AFFECT_READOUT` | `sample` | 情绪读出：`map` 取后验均值（稳定，消逐轮翻号）/ `sample` 逐轮采样（默认，带随机波动） |
| `ZERO_APPRAISE_CALIBRATE` | 关 | 分级标定锚抵消 LLM 把负面读太轻的正向偏置（敌意→更负）；`1` 开启。**视模型**——强模型（deepseek 类）本就够负、可不开 |
| `ZERO_EMOTION_NOISE_STD` | 0.05 | 每轮情绪的随机噪声幅度（调小=更稳、`0`=关该噪声源） |
| `ZERO_SAMPLE_SIGMA_MAX` | 0.5 | 后验采样的逐维抖动上限（仅 `sample` 读出下生效） |
| `ZERO_CHAT_RNG_SEED` | — | 固定随机种子，贯穿引擎采样 + 情绪噪声，便于 eval 复现（留空=每次随机） |
| `ZERO_EMOTION_BASELINE_ATTITUDE_W` | 0.6 | 情绪回落基线里「对此人态度」占比；`<1` 给回中性的拉力、防越聊越上头（`1`=旧行为） |

**② 治「越聊越暧昧 / 关系无端升温」**（科学家议会 seeking 吸引盆两轮裁决；默认全逐字零回归、荐值走注释，⭐=数字人推荐开）

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `ZERO_INTENSITY_FLOOR` | 0.2 | ⭐arousal 强度下限；设 `0` 去掉中性输入的正 arousal 直流底噪（暧昧滑移的根之一） |
| `ZERO_AROUSAL_BASELINE` | 0 | ⭐arousal 基准平移；负值（荐 -0.08）让平淡对话给零/负唤醒（副交感 vagal brake / deactivation） |
| `ZERO_ATTITUDE_REVERSION_A` | 同 valence(0.01) | ⭐态度 arousal 维**独立**回归率（荐 0.3–0.5）；令长期态度只累积效价、不累积唤醒偏置 |
| `ZERO_ATTITUDE_SETPOINT_A` | persona.setpoint[1] | 态度 arousal 回归锚；未设=取气质底色的 a、`0`=中性 |
| `ZERO_HABITUATION_TAU` | 关 | 习惯化 τ(轮，荐 5–10)：重复互动 arousal 响应按 `exp(-n/τ)` 递减（SCR 习惯化）；空/0=关 |
| `ZERO_AROUSAL_GAIN_CAP` | 不 cap | workspace `arousal_gain` 上限（荐 0.3–0.6）；防高唤醒正反馈失稳；空=旧无界 |
| `ZERO_ATTITUDE_RATE_DECAY_K` | 0 | 越熟态度形成越慢（关系止血 Q5-A）；`0`=关，仅减缓漂移、非真多稳态 |
| `ZERO_FAMILIARITY_TAU` | 20 | 熟悉度累积 τ(轮)，配合 `RATE_DECAY_K`；仅 `K>0` 时生效 |
| `ZERO_RELATIONSHIP_STAGE_HINT` | 关 | 给 LLM 关系距离软提示（曝光三档，关系止血 Q5-B；确定性派生、不经 LLM 判跃迁）；空/0=关 |

**③ 记忆 / 注意力窗 + 召回排序**（默认已按认知科学调好，一般不用动）

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `ZERO_HISTORY_WINDOW` | 40 | 喂 LLM 的工作记忆窗总条数（越大记越久、越费 token） |
| `ZERO_HISTORY_PRIMACY_K` | 5 | 窗内保留的「最初几条」（首因），其余留给最近几轮（近因） |
| `ZERO_RECALL_SIM_MIN` | 0.65 | 召回余弦相似度下限（越高越只在强相关时才想起旧事） |
| `ZERO_RECALL_INJECT_MIN` | 0.5 | 旧记忆升入注意力预算、与近期对话同台竞争的重要性门 ∈[0,1] |
| `ZERO_RECALL_DECAY_D` | 0.5 | 三维重排：recency 幂律衰减指数 d |
| `ZERO_RECALL_ALPHA` · `_BETA` · `_GAMMA` | 0.33 · 0.34 · 0.33 | 三维重排权重：recency · sim · importance |
| `ZERO_RECALL_IMPORTANCE_SCALE` | 30 | importance 归一 Hill 常数 C：`p/(p+C)` |
| `ZERO_RECALL_AROUSAL_MOD` | 0 | 唤醒调制召回 importance（`1` 开启） |
| `ZERO_EPISODE_MAX_PER_KEY` | 0（`--chat` 默认 300） | 单人情景记忆条数上限，满了删最旧（0=不限） |
| `ZERO_EPISODE_SALIENCE_MIN` | 0.15 | 情景写入的显著度门 `salience=precision×\|rpe\|`（含时间/约定内容旁路强写） |
| `ZERO_EPISODE_SALIENCE_AFFECTIVE_ADD` | 0 | 低唤醒高语义补偿 `salience+=0.3·\|value\|`（`1` 开启） |
| `ZERO_EPISODE_DEDUP_MAX` | 0.92 | 情景写入去重余弦阈（高于此视为近义跳过） |

> **其它进阶变量**（`.env.example` 未列）：`ZERO_CHAT_THREAD` 切对话线程 id，隔离不同会话的历史/态度/记忆 scope、防串味；`ZERO_PUSH_LOGIT_BIAS` 让 push 通路叠加 OpenAI `logit_bias`（需兼容 tokenizer，缺则优雅退回纯 prompt 用词倾向）。
> 想还原更早的行为逐项设回旧值即可（如窗口设回 `20`、`ZERO_EMOTION_BASELINE_ATTITUDE_W=1`）。

---

## 文档

- **[docs/](docs/README.md)** — 对外框架图 + 运作流程图（whiteboard-cli 渲染）
- **[DATASETS.md](DATASETS.md)** — 真网络化所需数据集清单（获取方式 / 许可）
- **[notes/](notes/)** — 研究笔记：情感数学、文本输出情绪、并行脑路与工作空间、数字人路线图
