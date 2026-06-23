# LLM 表达 vs 人类表达，以及「情感能否被数学拟似」——学术论证与推导

> **日期**：2026-06-23
> **性质**：一次学术性对话的成果固化（与本仓库「多 Agent harness」主题无关，作为独立研究笔记保留）。
> **生成方式**：基于 WebSearch / WebFetch 检索的同行评审文献整理，数学推导为标准结果的逐步展开。
> **核对说明**：文中文献依检索结果给出作者/年份/链接与核心结论；若用于正式论文/报告，建议按标题逐篇核实出处与具体数据。

---

## 目录

1. [为什么 LLM 的表达与真实人类表达存在系统性差异](#一为什么-llm-的表达与真实人类表达存在系统性差异)
2. [情感能否被现有数学工具「拟似」](#二情感能否被现有数学工具拟似)
3. [论文级推导 A：Gottman 婚姻方程的动力系统分析](#三推导-a-gottman-婚姻方程的完整动力系统分析)
4. [论文级推导 B：变分自由能 → 情感](#四推导-b-变分自由能从-jensen-不等式到-f--log-po-再到情感)
5. [完整文献来源](#五完整文献来源)

---

## 一、为什么 LLM 的表达与真实人类表达存在系统性差异

差异不是「还不够像」，而是**生成机制决定的、可观测、可度量的结构性偏移**。从五个层面论证。

### 1. 生成机制的根源：形式分布 vs. 交流意图
- **Bender & Koller (2020, ACL)《Climbing towards NLU》**：仅在文本形式（form）上训练，原则上学不到意义（meaning），意义需与外部世界、交流意图关联（grounding）。LLM 优化 $P(\text{token}_t\mid \text{token}_{<t})$ 的最大似然，本质是对语言形式的条件分布建模。
- **Bender, Gebru et al. (2021, FAccT)《On the Dangers of Stochastic Parrots》**：「随机鹦鹉」——按统计规律拼接语言形式，背后**没有 communicative intent**。
- 对照：人类表达是 **Grice (1975) 合作原则**下的意图驱动行为（言外之意 / 语用 implicature）；哲学上对应 **Searle (1980) 中文房间**——句法操作再完美不等于语义理解。
- **结论**：两者「生成目标函数」根本不同——一个逼近语料分布，一个达成交流目的。这是所有下游差异的源头。

### 2. 计量语言学：可统计的「机器指纹」
- **DetectGPT (Mitchell et al., 2023, ICML)**：机器文本倾向落在模型对数似然曲面的**负曲率区域**，人类文本不是。
- **困惑度与突发性（perplexity & burstiness）**：人类句长/复杂度起伏更大；LLM 更平滑、更可预测。
- **超额词汇（Kobak et al., 2024）**：PubMed 数百万摘要中，2023 年后 *delve, intricate, underscore* 等词频断崖式跃升，量级堪比 COVID 词频异动——LLM 用词偏好在真实语料中的可观测印记。
- **HC3 / Guo et al. (2023)**：LLM 文本词汇多样性（type-token ratio）更低、情感更中性、结构更模板化。

### 3. 对齐训练导致「分布收窄 / 模式坍缩」
- **Kirk et al. (2024, ICLR)**：RLHF 显著降低输出多样性（distribution sharpening），换取对齐与稳健性，输出趋向「安全、中庸、平均」。
- **谄媚（Sharma et al. 2023；Perez et al. 2022）**：模型迎合用户、回避鲜明立场；人类表达充满立场、偏见、矛盾。
- **模板化结构**：过度礼貌、开头铺垫、强行三段式 + 列表 + 总结——奖励模型偏好的副产物。
- 人类对话带不流畅（disfluency）、自我修正、犹豫、话题漂移；LLM 的「过度流畅」本身是非人类信号。

### 4. 语义/语用/具身的缺失
- 缺乏 grounding 与具身性：人类语言根植于感知运动经验、身体、社会互动；LLM 只有文本符号关系。
- 信息密度曲线不同：**均匀信息密度假说（UID, Levy & Jaeger）**——人类倾向均匀铺开信息（受工作记忆与增量产出约束）；LLM 是全局并行 attention，无「够用即可（good-enough）」的认知约束。
- 无真实 stake 与个人史：人类表达背后有利害、情绪、经历；LLM 的「第一人称」是统计平均的模仿。

### 5. 社会层面：同质化（homogenization）
- **Doshi & Hauser (2024, Science Advances)**：LLM 辅助写作提升个体创意，却降低群体层面多样性，长尾被削平。
- 去个性化：人类表达携带方言、社会身份、个体风格；LLM 趋向「无地域、无阶层、去个性」的平均语体。
- **Shumailov et al. (2024, Nature)** 的「模型坍缩」从另一侧印证：在自生成数据上递归训练会丢失分布尾部。

### 因果链小结
> 训练目标（似然最大化）→ 概率几何留下机器指纹 →（RLHF）分布进一步收窄、模板化 →（无 grounding/意图）语义与语用空心 →（规模化使用）社会表达同质化。

人类表达是**意图驱动、具身、个体化、信息增量产出**的；LLM 表达是**分布拟合、去具身、平均化、全局并行生成**的。差异是机制性的，可被语言学、计量统计、信息论、社会科学多路实证捕捉。

---

## 二、情感能否被现有数学工具「拟似」

**结论先行**：把「拟似」拆成五个递进任务——represent（表示）/ predict dynamics（预测动态）/ infer from causes（从成因推断）/ explain generation（解释生成）/ reproduce experience（复刻体验）。**第 1–4 级都有成熟且在进步的数学；第 5 级遇到原则性障碍。**

### 能拟似的四条路径（+1 条统计路径）

**路径一 · 几何/拓扑表示**
- Russell (1980) 环状模型（circumplex）：MDS/因子分析从相似性判断降维，稳定坍缩成二维且排成圆环。
- 数学：$e\approx(v,a)\in\mathbb{R}^2$；极坐标 $r=\sqrt{v^2+a^2}$（强度）、$\theta=\arctan(a/v)$（类别）。Mehrabian & Russell 的 **PAD** 加一维支配度成 $\mathbb{R}^3$。
- 维度模型常胜过离散分类（连续回归捕捉标签间渐变）。
- 拟似了「结构与度量」，未触及「怎么动 / 从哪来」。

**路径二 · 动力系统**
- 核心：情感是随时间演化的状态 $x(t)$。
- Gottman 婚姻数学：非线性差分方程 + 双线性影响函数，对「5 年后离婚/存续」约 94% 事后准确率（详见第三节）。
- 推广：Liebovitch 心理治疗动力系统；随机微分方程（SDE）建模情绪调节；时滞二元互动模型。
- 拟似了「时间演化、相互作用、稳定性」，可外推预测。

**路径三 · 概率/逻辑评价模型**
- Appraisal theory（Arnold/Lazarus/Scherer）：情感是认知评价的结果。
- **OCC (1988)**：22 类情感按「事件×可欲性 / 主体×可赞性 / 客体×吸引力」组织成逻辑树 + 强度函数；为计算可处理而设计。
- 形式化推进：多智能体仿射概率逻辑（AfPL）、JAIR 上 OCC 的概率形式化、透明模型 EEGS。
- 拟似了「认知因果结构」，可解释可生成；短板是规则脆性与文化依赖。

**路径四 · 自由能/主动推理**
- Seth & Friston：内感受推理——情感 = 内感受预测误差最小化；通过更新信念（重评）或改变身体（allostasis）调节。
- 变分自由能 $F=\mathbb{E}_Q[\log Q(s)-\log P(o,s)]\ge -\log P(o)$；精度加权预测误差是强度旋钮。
- circumplex × 自由能（arXiv 2407.02474）：效价 $V=U-EU$（奖励预测误差），唤醒 $A=H[Q(s\mid o)]$（后验熵）。
- 最有野心：给情感统一的生成式原理（详见第四节）。

**路径五（补充）· 纯统计/机器学习**
- 深度网络从面部/语音/文本/生理信号端到端回归 valence-arousal 或分类；LLM 涌现细粒度情感处理。
- 「算得动」，但借用前四条的标注体系，继承其效度问题。

### 三道学术上公认的鸿沟

**鸿沟一 · 测量效度：被建模的「情感」本体论有争议**
- **Barrett et al. (2019)《Emotional Expressions Reconsidered》**（PSPI）：情感不是可从面部可靠读取的普遍离散类；情感建构论（theory of constructed emotion）主张情感是大脑用概念在情境中建构的。
- 一批专家断言当前「情感识别」缺乏科学基础（ACLU）。
- 含义：拟合目标本身定义不清 = garbage in；离散 vs 维度的理论不一致至今未弥合。

**鸿沟二 · 表征坍缩：连续/模糊/流动被压成标签会丢信息**
- **《Modelling Emotions is an Elusive Pursuit》(arXiv 2603.23017)**：标注者完全一致仅约 20%；跨模态一致性仅 4.18%；情感在单句内不断漂移（熵尖峰=转换点）。
- 分歧不是噪声，而是情感固有的歧义与主观性；软标签/分布式表示仍不成熟，未真正解决。

**鸿沟三 · 本体论鸿沟：功能性情感 ≠ 被体验的感受质（qualia）**
- **Chalmers 的「意识难问题」**：为何/如何物理或计算过程会伴随主观体验，至今未解。
- 文献明确区分「人工情感=对刺激的计算响应」与「真实情感=带主观体验与意识」；前者不蕴含后者。
- 情感体验本身的内容恰是计算建模长期忽视的部分；务实派搁置 hard problem 仍能造出「有情感功能」的系统——恰反证现有数学是在搁置体验本身的前提下才「成功」。

### 综合判断

| 「拟似」的层级 | 代表数学工具 | 现状 |
| --- | --- | --- |
| 表示结构（坐标/几何） | circumplex、PAD、MDS/PCA | ✅ 成熟 |
| 预测动态（随时间演化） | Gottman 差分方程、SDE、时滞模型 | ✅ 成熟，可外推 |
| 从认知评价推断 | OCC + 概率逻辑 | ✅ 可生成、可解释 |
| 解释生理生成机制 | 自由能 / 主动推理 / 内感受推理 | ✅ 有统一变分框架 |
| 从信号自动识别 | 深度学习多模态、LLM | ⚠️ 算得出，效度受质疑 |
| 复刻主观体验（qualia） | —— | ❌ 原则性未解（hard problem） |

**哲学定位**：现有数学拟似的是情感的**代理变量与投影**（坐标、轨迹、概率、预测误差、信号特征），是 operationalized 模型，而非情感本体。三道鸿沟：鸿沟一质疑「靶子是否真实」，鸿沟二说明「投影会丢信息」，鸿沟三指出「投影够不到本体」。前两道是经验性困难（会随更好方法缓解），第三道是形而上学困难（不随算力消失）。

**一句话**：今天的数学能很好地**描述、预测、生成、干预**情感的功能后果（预测离婚、驱动 NPC、解释内感受、识别压力），但这些都是对情感**外在投影**的拟合；情感**作为被体验到的主观感受本身**目前无法被数学复刻——挡路的不是工具不够强，而是尚未解决的意识难问题。**能拟似「情感的影子」，还不能拟似「感受到情感」这件事本身。**

---

## 三、推导 A：Gottman 婚姻方程的完整动力系统分析

### A.1 未受影响模型与「惯性」的数学含义
单人自回归：$W_{t+1}=r_W W_t+a_W,\ 0<r_W<1$。
不动点 $W^\ast_{\text{uninf}}=\dfrac{a_W}{1-r_W}$（未受影响稳态 = 天然情绪基调）。
误差 $e_t=W_t-W^\ast$ 满足 $e_{t+1}=r_W e_t\Rightarrow e_t=r_W^{\,t}e_0$。$|r_W|<1$ 稳定；$r_W\to1$ 回归极慢——**情感惯性 = 线性化特征值趋近 1**。

### A.2 二维系统与双线性影响函数
$$
W_{t+1}=r_W W_t+a_W+I_{H\to W}(H_t),\qquad
H_{t+1}=r_H H_t+a_H+I_{W\to H}(W_t).
$$
双线性影响函数（带正负阈值 $T^\pm$，阈内为 0）：
$$
I_{H\to W}(H)=
\begin{cases}
\beta_W (H-T^+), & H>T^+\\
0, & T^-\le H\le T^+\\
\alpha_W (H-T^-), & H<T^-
\end{cases}
$$
经验上负侧更陡 $|\alpha_W|>|\beta_W|$（消极更具传染性），是「消极吸收态」的根源。

### A.3 相平面：零斜线与不动点
$W$ 零斜线（$\Delta W=0$）：$W=\dfrac{a_W+I_{H\to W}(H)}{1-r_W}$；同理 $H$ 零斜线。
不动点 = 两条零斜线交点。影响函数分段线性 ⇒ 零斜线是折线 ⇒ **可多次相交 = 多稳态**（线性模型做不到的质变）。

### A.4 稳定性：Jacobian
$$
J=\begin{pmatrix} r_W & I'_{H\to W}(H^\ast)\\ I'_{W\to H}(W^\ast) & r_H\end{pmatrix}.
$$
离散系统判据是**单位圆**：$|\lambda_{1,2}|<1$ 稳定吸引子；一个 $|\lambda|>1$ 为鞍点；非对角元就是影响函数斜率——影响越强越易改变相图拓扑。

### A.5 双稳态与「消极吸收态」
典型出现两个稳定吸引子 + 中间鞍点：
- 正–正象限：良性稳态；
- 负–负象限：消极吸收态（negative absorbing state）；
- 鞍点稳定流形 = 分界线 separatrix，切出两个吸引盆（basins）。

落到哪个稳态取决于初始条件在哪个吸引盆；而吸引盆大小由结构参数决定：基调阴郁（$a/(1-r)<0$）+ 高惯性 $r$ + 负侧斜率 $\alpha$ 陡 ⇒ 负吸引盆撑大 ⇒ 日常互动易越过 separatrix 滑入消极吸收态。「负向螺旋」= 相空间拓扑本身把夫妻往负稳态拽。

### A.6 为何短观测能「预测」长期（及折扣）
逻辑链：短观测 → 估结构参数 → 参数决定相图拓扑 → 拓扑决定渐近命运。约 94% 准确率源于此。
**诚实折扣**：94% 是已知结局样本上的事后分类（postdiction / 样本内），其前瞻预测的可复制性与过拟合风险受批评。模型作为动力学解释框架有力，但别把该数字当作开箱即用的预测精度。

---

### A.7 数值实例：一组参数下的双稳相图（纯标准库实算）

取对称、基调中性（$a_W=a_H=0$）的设定，影响函数用**有界 tanh 形**（Gottman 双线性影响函数的光滑/饱和变体——定性行为一致，关键是「有界」才能产生多稳）：
$$W_{t+1}=r\,W_t+A\tanh(kH_t),\qquad H_{t+1}=r\,H_t+A\tanh(kW_t),$$
参数 $r=0.5,\ A=0.6,\ k=2.0$。

**为何双稳（pitchfork 判据）**。对角解满足 $(1-r)s=A\tanh(ks)$。原点 $s=0$ 永远是解；当原点处 tanh 斜率超过直线斜率，即
$$Ak>1-r\quad(\text{此处 }1.2>0.5),$$
原点沿对角方向失稳，**叉式分岔（pitchfork）**生出两个对称非零稳态 $\pm s^\ast$。数值解得 $s^\ast=1.178681$。

**三个不动点与稳定性**（Jacobian 特征值，解析）：

| 不动点 $(W,H)$ | 特征值 $\lambda_1,\lambda_2$ | 类型 |
| --- | --- | --- |
| $(0,0)$ | $+1.700,\ -0.700$ | 鞍点（saddle）|
| $(+1.1787,+1.1787)$ | $+0.542,\ +0.458$ | 稳定吸引子 $P_+$ |
| $(-1.1787,-1.1787)$ | $+0.542,\ +0.458$ | 稳定吸引子 $P_-$ |

原点的不稳定方向特征向量 $(1,1)$（对角，$\lambda=1.7>1$），稳定方向 $(1,-1)$（反对角，$|\lambda|=0.7<1$）——故 **separatrix（分界线）≈ 反对角 $W=-H$**，正是它的稳定流形。

**吸引盆 ASCII 相图**（每个格点迭代 400 步看收敛到哪个吸引子；$W$ 横轴、$H$ 纵轴 $\in[-2,2]$）：

```
+---------------------------------------------------+
|.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAA@AAAAAAAAAAAA|
|BBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBSAAAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAA|
|BBBBBBBBBB#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAA|
|BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB.|
+---------------------------------------------------+
```
- `A` = 收敛到 $P_+$（双方积极），`B` = 收敛到 $P_-$（双方消极）；`@`=$P_+$，`#`=$P_-$，`S`=原点鞍点。
- 分界线沿反对角：初始情绪落在它上方 → 被吸入积极稳态，落在下方 → 被吸入消极稳态。这就是「**落入哪个吸引盆决定关系走向**」的字面图像。
- 把基调调阴郁（$a_W,a_H<0$）或令负侧影响更猛，会整体推移分界线、**扩大 B 盆**——消极吸收态的吸引域变大，日常波动更易滑入。

**附：高清矢量版脚本**（本地 `pip install numpy matplotlib` 后可跑出 nullclines + 流场 + 轨迹的 PNG）：
```python
import numpy as np, matplotlib.pyplot as plt
r, A, k = 0.5, 0.6, 2.0
infl = lambda x: A*np.tanh(k*x)
H = np.linspace(-2, 2, 400); W = np.linspace(-2, 2, 400)
W_null = infl(H)/(1-r)          # W-nullcline: ΔW=0
H_null = infl(W)/(1-r)          # H-nullcline: ΔH=0
ws, hs = np.meshgrid(np.linspace(-2,2,25), np.linspace(-2,2,25))
dW = r*ws + infl(hs) - ws; dH = r*hs + infl(ws) - hs
plt.figure(figsize=(6,6))
plt.streamplot(ws, hs, dW, dH, color='0.75', density=1.1)
plt.plot(W_null, H, lw=2, label='W-nullcline')
plt.plot(W, H_null, lw=2, label='H-nullcline')
s = 1.178681
plt.scatter([s,-s,0], [s,-s,0], c=['g','g','r'], s=90, zorder=5)
plt.plot([-2,2], [2,-2], 'r--', lw=1, label='separatrix W=-H')
plt.xlabel('W (wife)'); plt.ylabel('H (husband)')
plt.legend(loc='upper left'); plt.title('Gottman bistable phase portrait')
plt.savefig('gottman_phase.png', dpi=150, bbox_inches='tight')
```

---

## 四、推导 B：变分自由能（从 Jensen 不等式到 $F\ge-\log P(o)$，再到情感）

### B.1 问题：惊异算不出
最小化惊异 $-\log P(o)$，其中 $P(o)=\int P(o,s)\,ds$（模型证据）一般 intractable。

### B.2 Jensen 不等式 → 上界 $F$
引入近似后验 $Q(s)$：
$$
-\log P(o)=-\log\mathbb{E}_Q\!\left[\frac{P(o,s)}{Q(s)}\right]
\le \mathbb{E}_Q\!\left[-\log\frac{P(o,s)}{Q(s)}\right]=F,
$$
（$-\log$ 凸，Jensen）。得
$$
F=\mathbb{E}_Q[\log Q(s)-\log P(o,s)]\ \ge\ -\log P(o).
$$

### B.3 KL 分解 → 上界紧致性
用 $P(o,s)=P(s\mid o)P(o)$：
$$
F=D_{\mathrm{KL}}[Q(s)\,\|\,P(s\mid o)]-\log P(o),
$$
故 $F-(-\log P(o))=D_{\mathrm{KL}}[Q\|P(s\mid o)]\ge0$。间隙正是近似后验与真后验的 KL；$Q=P(s\mid o)$ 时取等。

### B.4 两个直觉分解
- complexity − accuracy：$F=D_{\mathrm{KL}}[Q(s)\|P(s)]-\mathbb{E}_Q[\log P(o\mid s)]$（奥卡姆剃刀）。
- energy − entropy：$F=-\mathbb{E}_Q[\log P(o,s)]-H[Q(s)]$。

### B.5 高斯近似 → 精度加权预测误差
生成模型 $o=g(s)+\omega_o,\ \omega_o\sim\mathcal N(0,\Pi_o^{-1})$；$s=\eta+\omega_s,\ \omega_s\sim\mathcal N(0,\Pi_s^{-1})$（$\Pi$=精度）。
$$
F\approx\tfrac12\varepsilon_o^\top\Pi_o\varepsilon_o+\tfrac12\varepsilon_s^\top\Pi_s\varepsilon_s-\tfrac12\log|\Pi_o\Pi_s|,\quad
\varepsilon_o=o-g(\mu),\ \varepsilon_s=\mu-\eta.
$$
- 感知：$\dot\mu\propto g'(\mu)^\top\Pi_o\varepsilon_o-\Pi_s\varepsilon_s$。
- 主动推理：$\dot a\propto-\left(\frac{\partial\varepsilon_o}{\partial a}\right)^\top\Pi_o\varepsilon_o$。
精度 $\Pi$ 加权：高精度误差放大（凸显/紧迫），低精度被忽略；由神经调质（多巴胺等）编码——情感强弱的旋钮。

### B.6 落到情感
- 内感受版本（Seth & Friston）：$o$=内感受信号，$s$=身体状态信念；情感=内感受预测误差最小化；调节走重评或 allostasis。
- circumplex × 自由能（arXiv 2407.02474）：
  - 效价 $V=U-EU$（奖励预测误差），$U=\log P(o_t\mid C)$，$EU=\mathbb{E}_{Q(o_t\mid s_{t-1},\pi)}[\log P(o_t\mid C)]$；
  - 唤醒 $A=H[Q(s\mid o)]$（后验熵=不确定性）；
  - 回到几何：$r=\sqrt{V^2+A^2}$，$\theta=\arctan(A/V)$。
  效价–唤醒平面被改写成自由能量纲下的**生成式坐标**。

### B.7 把 B.5 展开到分层预测编码的消息传递方程

把 B.5 的「单个精度加权误差单元」堆叠成层级，就得到 Friston 的**分层预测编码（hierarchical predictive coding）**。设层级 $i=1,\dots,N$，每层有**表征单元**（信念均值 $\mu^{(i)}$）与**误差单元** $\varepsilon^{(i)}$。

**分层生成模型**（连续时间，广义运动坐标 $\tilde{\,\cdot\,}=(\mu,\mu',\mu'',\dots)$）：
$$
\tilde v^{(i-1)}=g^{(i)}\!\big(\tilde x^{(i)},\tilde v^{(i)}\big)+\tilde z^{(i)},
\qquad
\dot{\tilde x}^{(i)}=f^{(i)}\!\big(\tilde x^{(i)},\tilde v^{(i)}\big)+\tilde w^{(i)} ,
$$
$x$=隐动态状态，$v$=因果状态（层间传递的「输出」），噪声精度 $\Pi_v^{(i)},\Pi_x^{(i)}$。最低层 $v^{(0)}$=感觉输入（情感场景里 = 内感受信号）。

**两类预测误差（每层）**：
$$
\varepsilon_v^{(i)}=\tilde v^{(i-1)}-g^{(i)}\!\big(\tilde\mu_x^{(i)},\tilde\mu_v^{(i)}\big),
\qquad
\varepsilon_x^{(i)}=\mathcal D\,\tilde\mu_x^{(i)}-f^{(i)}\!\big(\tilde\mu_x^{(i)},\tilde\mu_v^{(i)}\big).
$$

**精度加权（误差单元的增益）**：
$$
\xi_v^{(i)}=\Pi_v^{(i)}\,\varepsilon_v^{(i)},\qquad
\xi_x^{(i)}=\Pi_x^{(i)}\,\varepsilon_x^{(i)} .
$$

**识别动力学（对 $F$ 做广义梯度下降）**：
$$
\dot{\tilde\mu}_v^{(i)}=\mathcal D\,\tilde\mu_v^{(i)}
+\Big(\tfrac{\partial g^{(i)}}{\partial\mu_v^{(i)}}\Big)^{\!\top}\xi_v^{(i)}
+\Big(\tfrac{\partial f^{(i)}}{\partial\mu_v^{(i)}}\Big)^{\!\top}\xi_x^{(i)}
-\;\xi_v^{(i+1)} ,
$$
$$
\dot{\tilde\mu}_x^{(i)}=\mathcal D\,\tilde\mu_x^{(i)}
+\Big(\tfrac{\partial g^{(i)}}{\partial\mu_x^{(i)}}\Big)^{\!\top}\xi_v^{(i)}
+\Big(\tfrac{\partial f^{(i)}}{\partial\mu_x^{(i)}}\Big)^{\!\top}\xi_x^{(i)}
-\;\xi_x^{(i)} ,
$$
$\mathcal D$ 是广义坐标的移位算子（$\mathcal D\tilde\mu=(\mu',\mu'',\dots)$）。

**消息传递的方向（核心）**：
- **自上而下**：高层把预测 $g^{(i)},f^{(i)}$ 送下去，**压制**低层误差单元 $\varepsilon^{(i)}$；
- **自下而上**：低层把精度加权误差 $\xi^{(i)}$ 送上去，**驱动**高层表征更新（上式末项 $-\xi_v^{(i+1)}$ 即来自下层的上行误差）；
- **横向**：精度 $\Pi^{(i)}$ 调节每层误差增益 = **注意 / 凸显**旋钮。

**落到情感（内感受层级）**：
- 最低层 $v^{(0)}$=内感受信号（心率、内脏、化学感受）；上层 $x,v$=对身体/生理原因的信念。
- **效价**对应内感受预测误差的符号性整合（「比预期好/坏」，与 B.6 的 $V=U-EU$ 同源）；**唤醒**对应被赋予高精度 $\Pi$ 的内感受误差的幅度/不确定性（与 B.6 的 $A=H[Q]$ 同源）。
- **主动推理在此层 = 自主神经/内分泌反射**：高层把内感受预测当作**设定点（set-point）**下行，身体通过反射改变状态来抹平 $\varepsilon$（allostasis）——即「用行动而非更新信念来消除内感受预测误差」。
- 情感失调（焦虑/述情障碍等）在这套方程里被解释为**精度配置异常**：内感受误差精度过高/过低 → $\xi$ 失衡 → 信念与自主调节被错误驱动。

一句收束：B.5 是「单个误差单元」，B.7 是「误差单元的层级网络」；情感不是某一层的产物，而是**整条内感受预测层级上精度加权误差的全局最小化轨迹**。

### 两条推导的会合点
- Gottman：情感的人际动态进相空间，命运 = 吸引子拓扑。
- 自由能：情感的生成机制进变分推断，感受 = 精度加权预测误差。
- 两者都把情感化成可计算代理量（轨迹 / 误差），但都止步于第三鸿沟：相图里没有「被感受到的痛苦」，自由能方程里 $\varepsilon$ 再大也不自动等于「有东西在疼」。**数学能写出情感的运动方程，却写不出方程被体验的那一刻。**

---

## 五、完整文献来源

**LLM vs 人类表达**（依记忆给出，建议核实）
- Bender & Koller (2020, ACL)《Climbing towards NLU》；Bender, Gebru et al. (2021, FAccT)《Stochastic Parrots》
- Mitchell et al. (2023, ICML) DetectGPT；Kobak et al. (2024) excess vocabulary in PubMed；Guo et al. (2023) HC3
- Kirk et al. (2024, ICLR) RLHF & diversity；Sharma et al. (2023) / Perez et al. (2022) sycophancy
- Levy & Jaeger（UID 假说）；Searle (1980) 中文房间；Grice (1975) 合作原则
- Doshi & Hauser (2024, Science Advances)；Shumailov et al. (2024, Nature) model collapse

**情感的数学拟似**（含链接）
- 表示：[情感分类综述 (Wikipedia)](https://en.wikipedia.org/wiki/Emotion_classification) · [Valence-Arousal Space](https://www.emergentmind.com/topics/valence-arousal-space) · [Affective Computing 系统综述](https://arxiv.org/pdf/2203.06935)
- 动力系统：[The Mathematics of Marriage (ResearchGate)](https://www.researchgate.net/publication/232424148_The_Mathematics_of_Marriage_Dynamic_Nonlinear_Models) · [Gottman 原始论文 PDF](https://www.johngottman.net/wp-content/uploads/2011/05/The-Mathematics-of-Marital-Conflict-Dynamic-Mathematical-Nonlinear-Modeling-of-Newlywed-Marital-Interaction.pdf) · [Bilinear Influence Function (Chronicle)](https://www.chronicle.com/article/every-unhappy-family-has-its-own-bilinear-influence-function/) · [心理治疗动力系统 (Frontiers 2023)](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2023.980739/full) · [SDE 情绪调节 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6433426/) · [时滞二元互动模型](https://arxiv.org/pdf/1202.2338)
- 评价模型：[人工情感建模综述 (Frontiers 2016)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00021/full) · [OCC 概率形式化 (JAIR)](https://jair.org/index.php/jair/article/view/11052) · [agent 情感逻辑](https://www.sciencedirect.com/science/article/abs/pii/S1389041724000755) · [EEGS 透明情感模型](https://arxiv.org/pdf/2011.02573)
- 自由能/主动推理：[内感受推理与情感脑 (Phil. Trans. R. Soc. B 2016)](https://royalsocietypublishing.org/doi/10.1098/rstb.2016.0007) · [内感受/情感预测编码 (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3940887/) · [自由能原理 (Wikipedia)](https://en.wikipedia.org/wiki/Free_energy_principle) · [FEP for Perception and Action](https://arxiv.org/pdf/2207.06415) · [自由能 × 环状模型 (arXiv 2024)](https://arxiv.org/html/2407.02474v1)
- 统计/ML：[多模态压力检测](https://arxiv.org/pdf/2508.10468) · [VA 数据集综述](https://arxiv.org/pdf/2510.00738) · [LLM 细粒度情感处理](https://arxiv.org/pdf/2309.01664)
- 鸿沟一（效度）：[Barrett et al. 2019 (SAGE)](https://journals.sagepub.com/doi/10.1177/1529100619832930) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/31313636/) · [情感识别缺乏科学基础 (ACLU)](https://www.aclu.org/news/privacy-technology/experts-say-emotion-recognition-lacks-scientific)
- 鸿沟二（坍缩）：[Modelling Emotions is an Elusive Pursuit (arXiv 2603.23017)](https://arxiv.org/html/2603.23017)
- 鸿沟三（qualia）：[Cambridge Handbook Ch.30](https://www.cambridge.org/core/books/abs/cambridge-handbook-of-computational-cognitive-sciences/computational-models-of-emotion-and-cognitionemotion-interaction/42821F345649A9595695D6C7DAF5BACC) · [Modeling the Experience of Emotion](https://arxiv.org/pdf/0903.0735) · [机器意识的情感模型](https://arxiv.org/pdf/1701.00349) · [Detecting Qualia](https://arxiv.org/pdf/1712.04020)
