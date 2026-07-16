"""编排层共享状态：Stimulus 输入与 AffectState。

AffectState 用 pydantic 定义结构化 state；节点只返回增量字段（见 orchestration-rules.md）。
state 不放大对象（向量/文档）；trace 仅存标量中间量。运行态字段（value_table、
后验、采样点）由 Checkpointer 持久化，不写入长期记忆图谱。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from src.memory.types import Fact
from src.orchestration.external_prior import ExternalPrior


class Stimulus(BaseModel):
    """一个待评价的事件（OCC 评价输入）。各评价维度取值约定在 [-1, 1]。

    用 pydantic 模型以便被 LangGraph Checkpointer 原生序列化。构造用关键字参数。
    """

    name: str
    text: str | None = None  # 文本型 stimulus 原始文本；文本路径用，OCC 路径忽略
    goal_congruence: float = 0.0  # 与目标的一致性（事件维度）
    standard_compliance: float = 0.0  # 与标准的契合（行为维度）
    attitude_appeal: float = 0.0  # 对象的喜好（吸引力维度）
    intensity: float = 1.0  # 事件显著度/强度
    # 情境控制感评价维度（议会 2026-07-13 T2；Smith & Ellsworth 1985 control 维）。
    # 独立于 goal_congruence——目标可实现但感觉无法掌控（如意外之喜）。
    # +1=高控制/趋近（愤怒端），-1=低控制/回避（恐惧端）。
    # None=absent cue（B3：absent cue 精度趋零，不参与 B3 融合）；
    # 0.0=genuine-zero（显式真中性，参与 B3 融合，但贡献极小）。
    control_appraisal: float | None = None


class AffectState(BaseModel):
    """情感流水线全程共享的状态。节点只返回增量字段。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 输入
    stimulus: Stimulus | None = None

    # Perception
    features: list[float] = Field(default_factory=list)
    text_affect: tuple[float, float] | None = None  # 文本(v,a)；PerceptionAgent写、AffectCore读

    # Appraisal（OCC 理性先验）
    appraisal: dict[str, float] = Field(default_factory=dict)
    prior_mu: tuple[float, float] | None = None
    prior_sigma: tuple[float, float] | None = None
    reward: float | None = None

    # Value（在线 TD，价值/精度辅佐）—— value_table 是运行态
    value_table: dict[str, float] = Field(default_factory=dict)
    value_estimate: float | None = None
    rpe: float | None = None
    precision: float | None = None

    # AffectCore（主动推断，后验 + 采样）
    post_mu: tuple[float, float] | None = None
    post_sigma: tuple[float, float] | None = None
    affect_sample: tuple[float, float] | None = None  # e*（随机性来源）
    affect_precision: float | None = None  # 后验精度（工作空间再入用：内核 vs 语言的投票权）
    ignited_streams: list[str] = Field(default_factory=list)  # 工作空间点燃的并行流名（可观测）

    # Regulation / Expression（双通路·多通道）
    regulated_affect: tuple[float, float] | None = None
    expression: dict[str, Any] = Field(default_factory=dict)

    # 观测与作用域：trace 用 reducer 累加，节点只需返回自己的 [entry]（避免每步全量拷贝）
    trace: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    session_id: str = "default-session"
    user_id: str = "default-user"
    group_id: str = "default-group"

    # Mood（A.7 慢变心境：时间深度/滞后）—— 运行态，进 Checkpointer，不入图谱
    mood: tuple[float, float] | None = None

    # coping_potential 独立标量流（议会 2026-07-13；运行态慢变量）—— 进 Checkpointer，绝不入图谱
    # coping_potential_state: 情境控制感 ∈ [-1,1]（同 mood 先例：Checkpointer 持久，非图谱）
    # +1=趋近/高控制/愤怒端，-1=回避/低控制/恐惧端；经 language._appraisal_summary 消费
    # coping_potential_enabled=False 门控关 → 现路径逐字不变（零回归）
    coping_potential_state: float = 0.0
    coping_potential_enabled: bool = False  # 总门控（默认关=零回归）
    # facs_extended 扩展 AU 门控（设计门 PASS·路径 b；默认关=零回归）
    # True → ExpressionAgent 占位路径把 coping_potential_state 透传给 decode_channels，
    # 启用 11-AU 扩展集合（FACS_KEYS_EXT）；False=旧 5-AU 逐字行为（零回归）。
    # 经 chat_driver 读 ZERO_FACS_EXTENDED → SessionConfig → to_state_flags 贯通。
    facs_extended: bool = False  # 默认关=零回归
    # voluntary_coping_leak：双通路差异化（议会 C1 设计门 2026-07-14）∈[0,1]。
    # 自发头(push·锥体外路·皮层下驱动)全量传 coping；随意头(pull·锥体束意志调控)传
    # coping×voluntary_coping_leak（意志可部分压制 coping-driven AU，Rinn 1984）。
    # 默认 1.0 = 两头等值 = 逐字旧行为（零回归）；推荐 0.3；经 ZERO_VOLUNTARY_COPING_LEAK 注入。
    # 仅在 facs_extended=True 时对 facs_au 生效（legacy 模式 coping 不参与→自动零回归）。
    voluntary_coping_leak: float = Field(default=1.0, ge=0.0, le=1.0)

    # HPA/皮质醇慢回路（P3 1-B；运行态慢变量）—— 进 Checkpointer，**绝不写入长期记忆图谱**
    # cortisol_state: 归一皮质醇水平 ∈ [0, 1]（同 mood 先例：Checkpointer 持久，非图谱）
    # cortisol_updated_at: 上次更新的 UTC epoch 秒（float，非 datetime——msgpack 原生/无时区歧义）
    # cortisol_enabled=False 门控关 → 现路径逐字不变（零回归）
    cortisol_state: float = 0.0
    cortisol_updated_at: float | None = None  # UTC epoch 秒；None=首次未初始化
    cortisol_enabled: bool = False  # 总门控（默认关=零回归）
    cortisol_arousal_gate: bool = False  # cortisol→arousal 基线抬升（默认关=零回归）
    cortisol_attitude_gate: bool = False  # cortisol→ATTITUDE_RATE 放大（默认关=零回归）
    cortisol_arousal_alpha: float = 0.0  # 默认 0 → offset=0 → 零回归
    cortisol_attitude_alpha: float = 0.0  # 默认 0 → rate_eff=ATTITUDE_RATE × 1 → 零回归
    # 动力学常数（None → AppraisalAgent 回退 affect_math 议会推荐常量=零回归；env 设值则覆盖）
    cortisol_tau: float | None = None  # 衰减 τ 秒；None→CORTISOL_TAU_DECAY(≈5400)
    cortisol_impulse: float | None = None  # 单次应激脉冲量；None→CORTISOL_IMPULSE
    cortisol_theta_goal: float | None = None  # 触发目标阈；None→CORTISOL_THETA_GOAL
    cortisol_theta_intensity: float | None = None  # 触发强度阈；None→CORTISOL_THETA_INTENSITY

    # P3 1-C ToM / 社会情绪：对方情绪 VAD 估计（图外 appraise_text 确定性词典法产出后注入；
    # 非自身 OCC goal_congruence——语义独立、字段独立，同 recalled_disposition 标量注入模式）。
    # None = 门控关 = 零回归（不影响自身先验）；估计在 chat_driver 图外，不进确定性节点内部。
    interlocutor_affect: tuple[float, float] | None = None
    # 情绪传染 w_c（差分式·双维；默认 0=零回归；硬上界 ≤0.3 由 .env 注释约束）
    contagion_alpha: float = 0.0
    # CARE 动机先验抬高（对方 v_i<0 触发；默认 0=零回归；推荐 ≤0.4）
    care_bias_alpha: float = 0.0
    # 替代喜悦（对方 v_i>threshold 且 a_i>0 触发·v1 只做 joy；默认 0=零回归；推荐 ≤0.2）
    vicarious_alpha: float = 0.0
    # 替代喜悦触发阈（默认 0.3；与 care_bias 互斥触发：v<0 走 CARE、v>threshold 走替代）
    vicarious_threshold: float = 0.3

    # MemoryRecall：长期 user 情绪倾向回灌（MemoryRecallAgent 读 → AppraisalAgent 用）
    recalled_disposition: float | None = None
    # 语义召回（Graphiti 等语义记忆侧信道）的相关事实串 → 喂 LanguageAgent 检索；
    # 无语义后端时恒空（零回归）。仅存短文本，不放大对象（遵守 state 约束）。
    recalled_context: list[str] = Field(default_factory=list)
    # D1+D3：召回的原始 Fact（已三维重排，含 sim/valid_at/importance 线索），供 chat_driver
    # 按 importance 注入 history 预算竞争。Fact 全为标量/str/datetime，非大对象（满足 state 约束）；
    # 无语义后端时恒空（零回归）。仅 chat 驱动消费，不进 LLM 数学。
    recalled_facts: list[Fact] = Field(default_factory=list)

    # Language（affect↔language 双向收敛回路）—— 运行态观测量
    language_text: str | None = None  # 生成的语言内容
    language_affect: tuple[float, float] | None = None  # 语言反推出的情感
    language_consistency: float | None = None  # dist(language_affect, e*)，可观测
    language_iter: int = 0  # 回路迭代计数

    # 控制开关
    regulation_enabled: bool = False  # 开启掩饰/再评价（双通路对比）
    regulation_strategy: str = "suppression"  # 调节策略：suppression（默认）/ reappraisal（Gross）
    mood_enabled: bool = False  # 开启慢变心境的历史依赖/滞后（A.7）
    recall_enabled: bool = False  # 开启长期倾向回灌偏置 appraisal（记忆读闭环）
    language_enabled: bool = False  # 开启语言层 + affect↔language 双向收敛回路
    workspace_enabled: bool = False  # 开启显著度门控全局工作空间（并行流竞争+ignition+精度再入）
    appraisal_conditioning_enabled: bool = False  # 把 OCC 评价结构并入语言生成（CPM/EMA）
    language_max_iters: int = 3  # 回路终止上限（防死循环）
    rng_seed: int | None = None  # 采样可控（测试用）
    # 后验采样 sigma 上限（情绪「防抖」旋钮）：None → 用 affect_math.MAX_SAMPLE_SIGMA（零回归）。
    # 调小使 e* 样本更贴 post_mu、降低逐轮抖动；由 chat_driver 读 ZERO_SAMPLE_SIGMA_MAX 注入。
    sample_sigma_cap: float | None = None
    # 情绪读出模式（P4 议会议决，数学+神经一致）：'sample'=逐轮后验采样（默认·逐字旧行为）；
    # 'map'=取后验均值 e*=post_mu（MMSE 最优点估计），消除单样本大方差致的逐轮翻号，时序连续性
    # 交既有 emotion_decay_step 的 AR1≈0.4 承担（OU 离散化）。由 ZERO_AFFECT_READOUT 注入。
    affect_readout: str = "sample"
    # arousal 证据基准平移（P1-c 议会 Q7）：默认 0.0=旧整流行为（arousal 恒正）；负值启用
    # deactivation（平淡低强度输入可给零/负 arousal，副交感臂）。AppraisalAgent 读 → occ_prior，
    # 由 chat_driver 读 ZERO_AROUSAL_BASELINE 注入。仅标量，不入 LLM 数学（守热路径红线）。
    arousal_baseline: float = 0.0
    # arousal_gain 增益上限（P4-d 议会二轮·廉价 cap 防御）：默认 None → 不 cap（旧线性无界行为，
    # 零回归）；设 x∈[0.3,0.6] → arousal_gain 无条件钳到 1+x（高唤醒段效果最显著），
    # 防正反馈无界放大。完整倒 U 立项排后（值得升级的实测触发 arousal>0.5）。仅 workspace 路径用；
    # 标量、不入 LLM 热路径。
    arousal_gain_cap: float | None = None
    # precision_split（A-P1-A 议会裁决）：True 时 value 流证据精度用 precision_da(rpe)（消 β·V，
    # DA 路径仅由 |δ| 决定精度）；False=默认旧行为（零回归）。仅 workspace 路径用。
    precision_split: bool = False
    # fuse_independence_correct（A-P0-B 议会裁决）：True 时 value 流 valence 维精度置极小
    # （MIN_PRECISION），保留其 arousal 维精度；消去与 appraisal/survival 共线的 valence 冗余、
    # 防后验过度自信。False=默认旧行为（零回归）。仅 workspace 路径用。
    fuse_independence_correct: bool = False
    # ignition_survival_fallback（张力 4 裁决·神经席 M3）：True 时全弱刺激（无流过阈）改以
    # survival 流作兜底广播（皮层下生存信号始终有输出，与 GNW 亚阈=停留局部一致）；
    # 若 streams 无 survival 流则退回 max by salience（不崩）。
    # False=默认旧行为（全弱刺激保留 salience 最高流，零回归）。仅 workspace 路径用。
    ignition_survival_fallback: bool = False
    # ignition_beta（议会 2026-07-02 Item 2 · 软门控陡度）：None=硬 step 零回归（默认）；
    # 非 None → ignite() 用 logistic gate(sᵢ;θ,β)=σ(β·(sᵢ−θ)) 调制各流精度（连续近似
    # GNW all-or-none）。推荐区间 [20,50]，典型值 20；β<1 无意义。仅 workspace 路径用。
    # 由 ZERO_IGNITION_BETA env → chat_driver/runner → 此字段 → affect_core 透传。
    ignition_beta: float | None = None
    # P3 层级预测编码（1-A HPC v1）：默认 1=平层零回归（等价现 fuse_terms），2=启用 2 层 HPC。
    # 仅 workspace 路径的 fuse_terms 调用点消费；非 workspace 路径不触及（守零回归）。
    # 由 ZERO_HPC_LAYERS env → chat_driver/runner → 此字段 → affect_core 透传。
    hierarchical_layers: int = 1
    # P3 层间耦合强度 w∈[0,1]（HPC v1）：0=退平层（coupling=0 → 退化 fuse_terms(all)=零回归）；
    # 推荐 [0.3,0.8]；>1 hardcode raise ValueError（破坏凸组合稳定性，类 seeking 双稳坑）；
    # coupling 是 env 注入的**固定浮点标量**，非状态函数（禁 learning / 动态更新 / 运行期自适应）。
    # 由 ZERO_HPC_COUPLING env → chat_driver/runner → 此字段 → affect_core 透传。
    hierarchical_coupling: float = 0.0
    # mood_precision（悬而未决 T6-a · mood 流精度加权旋钮）：默认 0.8=MOOD_PRECISION 常量，
    # 介于主评价流与 SURVIVAL_PRECISION(0.4) 之间合理（Friston 2009 初始固定精度）。
    # 由 ZERO_MOOD_PRECISION env（默认注释关）→ chat_driver/runner → 此字段 → affect_core 透传。
    # 默认值=现常量 → 逐字零回归（不改行为）；可调 mood 流投票权。
    mood_precision: float = 0.8
    # text_affect_precision（悬而未决 T6-b · 文本语义流精度旋钮）：默认 0.3=TEXT_AFFECT_PRECISION
    # 常量（固定低值，显式低于 occ_prior 动态精度；Friston 2009 初版固定精度合理起点）。
    # 由 ZERO_TEXT_AFFECT_PRECISION env（默认注释关）→ chat_driver/runner → 此字段
    # → affect_core 透传。
    # 默认值=现常量 → 逐字零回归（不改行为）；未来可从回归器残差动态估计时替换此旋钮。
    text_affect_precision: float = 0.3
    # C（A-P2-A）：va_coupling 非对称系数。None → AppraisalAgent 调 occ_prior 时
    # 用默认 0.6/0.6（零回归）；非 None → 传入，启用 Kuppens 2013 negativity bias。
    # 经 build_chat_driver/runner → state → AppraisalAgent → occ_prior 贯通。
    va_coupling_pos: float | None = None
    va_coupling_neg: float | None = None
    # E（WARN-4 / A-P1-D）：Panksepp RAGE/FEAR 次级区分开关。默认 False → (-v,+a) 一律 rage
    # （零回归）；True → motivational_system 按 arousal 阈值在该象限分 fear/rage。仅
    # _appraisal_summary（appraisal_conditioning 开时）消费；精确阈值/坐标待议会 P1-D。
    panksepp_distinguish_fear: bool = False
    task_complete: bool = False

    # 外部多模态先验流注入口（议会 2026-07-15 M1–M6；PRP 外部多模态先验流注入口）。
    # 每条 ExternalPrior = (name, (μv, μa), (Πv, Πa))，逐维精度与 affect_core.py:77
    # streams 类型原生一致，AffectCoreAgent 展开后直接 extend 进 streams 竞争融合。
    # ⚠ 无任何图节点写此字段——每轮经 state_overrides 注入（interlocutor_affect 先例）；
    # 进 Checkpointer 不入图谱；默认空列表=零回归（workspace_enabled 下才有意义）。
    external_priors: list[ExternalPrior] = Field(default_factory=list)
    # 单条外部先验精度上界（M3；ZERO_EXTERNAL_PRIOR_PRECISION_CAP env；默认 0.8）。
    # 超出 → expand_external_priors raise ValueError（fail-fast 指向 MCP 传参错误）。
    # 经 SessionConfig → to_state_flags 贯通；MCP 保守低精度推荐见 design.md §五。
    external_prior_precision_cap: float = Field(default=0.8, gt=0.0)
    # 每轮外部流数上界（M6；ZERO_MAX_EXTERNAL_STREAMS env；默认 5）。
    # len(external_priors) > max → expand_external_priors raise ValueError（数学席 Σπ 虚增保险）。
    # 经 SessionConfig → to_state_flags 贯通；N≤5 保守（fuse_terms MIN_SIGMA 已给隐式上界）。
    max_external_streams: int = Field(default=5, ge=0)

    # text_coping 独立标量流（议会 2026-07-16 B3；来源：PerceptionAgent 词典/回归产出）。
    # ── text_coping_enabled：B3 总门控（默认 False=零回归）──
    # False → AppraisalAgent B3 强制 text=None，只走分支1/3（纯 ctrl 路径），逐字旧行为；
    # True → 允许 text_coping_prior 参与 B3 融合（分支2/4）。
    # 经 SessionConfig.text_coping_enabled → to_state_flags() → ainvoke 贯通。
    text_coping_enabled: bool = False  # 默认关=零回归
    # ── text_coping_prior：独立标量流入口 ──
    # None=门关=零回归（每轮归零防 LastValue 残留，仿 external_priors）；
    # 非 None=本轮 PerceptionAgent 产出的文本 coping 估计 ∈ [-1,1]。
    # 绝不入 fuse_terms/occ_prior/fast_survival_prior/hierarchical_fuse（来源正交）；
    # 唯一消费者经 AppraisalAgent 更新 coping_potential_state；每轮 step() 归零。
    text_coping_prior: float | None = None
    # ── text_coping_source：AppraisalAgent 输出 flag ──
    # True=本轮 coping_potential_state 的值来自 text（供 motivational_system 中间带哑火）；
    # False=来自 control_appraisal 或两皆 None（默认）；每轮 step() 归零。
    text_coping_source: bool = False
    # ── text_coping_precision：π_t（议会定 ≤0.10·缺省 0.08）──
    # B3 融合精度：两者皆有时 π_ctrl=1.0（固定）vs π_t（此字段）做精度加权。
    # ⚠ 不加 le 约束——避免 pydantic checkpoint 反序列化 fail（le 由 SessionConfig 层 fail-fast）。
    # 仿 text_affect_precision 注释风格；由 ZERO_TEXT_COPING_PRECISION env 注入。
    text_coping_precision: float = 0.08
