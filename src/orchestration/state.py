"""编排层共享状态：Stimulus 输入与 AffectState。

AffectState 用 pydantic 定义结构化 state；节点只返回增量字段（见 orchestration-rules.md）。
state 不放大对象（向量/文档）；trace 仅存标量中间量。运行态字段（value_table、
后验、采样点）由 Checkpointer 持久化，不写入长期记忆图谱。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.memory.types import Fact
from src.orchestration.external_prior import ExternalPrior


class Stimulus(BaseModel):
    """一个待评价的事件（OCC 评价输入）。各评价维度取值约定在 [-1, 1]。

    用 pydantic 模型以便被 LangGraph Checkpointer 原生序列化。构造用关键字参数。
    """

    name: str
    text: str | None = None  # 文本型 stimulus 原始文本；文本路径用，OCC 路径忽略
    # 四个评价维度的 [-1,1] 约束（议会 2026-07-28 第四轮 A4）：此前只有上面 docstring
    # 声明域，运行期不校验。后果实测：|intensity|≥1.2 时 occ_prior 的 conf 撞 1.0 →
    # σ 撞 MIN_SIGMA=0.05 → Π 撞死 800，比域内合法上限 200 高 4 倍，appraisal 流在
    # fuse_terms 里直接吃掉全部权重。与 expand_external_priors 的 M7 μ 域校验同类漏洞、同种修法。
    goal_congruence: float = Field(default=0.0, ge=-1.0, le=1.0)  # 与目标的一致性（事件维度）
    standard_compliance: float = Field(default=0.0, ge=-1.0, le=1.0)  # 与标准的契合（行为维度）
    attitude_appeal: float = Field(default=0.0, ge=-1.0, le=1.0)  # 对象的喜好（吸引力维度）
    intensity: float = Field(default=1.0, ge=-1.0, le=1.0)  # 事件显著度/强度
    # 情境控制感评价维度（议会 2026-07-13 T2；Smith & Ellsworth 1985 control 维）。
    # 独立于 goal_congruence——目标可实现但感觉无法掌控（如意外之喜）。
    # +1=高控制/趋近（愤怒端），-1=低控制/回避（恐惧端）。
    # None=absent cue（B3：absent cue 精度趋零，不参与 B3 融合）；
    # 0.0=genuine-zero（显式真中性，参与 B3 融合，但贡献极小）。
    control_appraisal: float | None = None
    # 域轴：当前事件的体裁/情境类别（B2·议会 2026-07-20；正交于 control_appraisal 方向轴）。
    #
    # 语义：
    #   None（默认）= absent/未指定 → 域门整体旁路，现有路径逐字不变（零回归）。
    #   "confrontational" = anger home 域（Twitter 式当下对抗·SemEval anger LB=0.857）。
    #   "survival_narrative" = fear home 域（ED 式生存/防御叙事·ED fear LB=0.90）。
    #   "neutral" = 显式两不属 → text_coping_prior 弃权（≠ None 旁路；两者语义不同·A-W1）。
    #
    # A-C2（对称门工程注意事项·神经层次不对等）：
    #   confrontational 域的 anger 激活依赖前额叶评价（可行动性·Harmon-Jones 2003
    #   DOI:10.1016/S0191-8869(02)00313-6），属皮层依赖过程；survival_narrative 域的 fear
    #   激活依赖皮层下防御回路（CeA-PAG·Davis&Walker 2009 DOI:10.1038/npp.2009.109），
    #   属皮层下生存机制。"对称门"是工程近似，两者神经层次不对等——anger 需语境中
    #   行动可行性、fear 则情境依赖性弱得多。
    #   重要：CeA/BNST 的解剖标签在此是**机制假说非确立事实**；Grogans et al. (2023 SCAN)
    #   meta 分析显示 CeA/BNST 无显著区域×确定性解离（效度不足）——故域定义以体裁/情境为锚，
    #   不以解剖结构为一级依据。
    #   domain_confidence（c_domain 精度衰减字段）属 B 类议会定，本轮不实施。
    #
    # None → 旁路；neutral → 显式弃权；domain_confidence 属 B 类（Feldman&Friston 2010 B 类）。
    domain: Literal["confrontational", "survival_narrative", "neutral"] | None = None

    @model_validator(mode="after")
    def _check_domain_ctrl_sign(self) -> Stimulus:
        """边界层 fail-fast（A-bd·议会 2026-07-20 T-1）：domain 非 None 且 control_appraisal
        非 None 时，校验二者符号一致性，防不一致注入抵达 appraisal 分支4 MLE。

        confrontational → ctrl >= 0（anger home 域不接受 fear 符号注入）；
        survival_narrative → ctrl <= 0（fear home 域不接受 anger 符号注入）；
        neutral / domain=None / ctrl=None → no-op，直接 return self（零回归）。

        违反时抛 ValueError（fail-fast，非静默）；消息含 domain 与 ctrl 值供调试。
        """
        domain = self.domain
        ctrl = self.control_appraisal
        if domain is None or ctrl is None:
            return self
        # neutral → no-op（跳过两 if·self 直接返回·防误加 neutral 的 ctrl 校验）
        if domain == "confrontational" and ctrl < 0.0:
            raise ValueError(
                f"domain={domain!r} 要求 control_appraisal >= 0，"
                f"但收到 ctrl={ctrl}（fear 符号注入 confrontational 域·边界层拒绝）"
            )
        if domain == "survival_narrative" and ctrl > 0.0:
            raise ValueError(
                f"domain={domain!r} 要求 control_appraisal <= 0，"
                f"但收到 ctrl={ctrl}（anger 符号注入 survival_narrative 域·边界层拒绝）"
            )
        return self


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
    # 启用 13-AU 扩展集合（FACS_KEYS_EXT）；False=旧 5-AU 逐字行为（零回归）。
    # 经 chat_driver 读 ZERO_FACS_EXTENDED → SessionConfig → to_state_flags 贯通。
    facs_extended: bool = False  # 默认关=零回归
    # canonical_physiology：physiology 占位口径门控（议会 2026-07-23·默认关=零回归）
    # True → 占位出 canonical {hr[50,120]/sc μS[0,20]/temperature_c[33,36]}；
    # False → legacy {hr[70,110]/sc[0,1]/pupil_mm[3,5]}，逐字零回归。
    # 经 ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER → chat_driver → SessionConfig → to_state_flags。
    canonical_physiology: bool = False  # 默认关=零回归
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

    # ── 精度量纲齐次化总门控（议会 2026-07-28 第四轮；ZERO_PRECISION_COMMENSURABLE env）──
    # False（默认）→ survival/mood/text/value 四条流沿用原裸常数与裸 sigmoid，**三条融合分支
    #   （gaussian_fuse 默认路径 / mood_enabled / workspace）全部逐字旧行为**；
    # True → 四条流的 Π 一律改写成 1/σ²、σ 表达在 [-1,1] 值域上（见 affect_math.py 齐次化节）。
    # ⚠ 与其它门控的关键区别：本门影响**每轮无条件执行**的 gaussian_fuse 默认路径
    #   （affect_core.py:149），不是只影响默认关的 workspace 分支。
    # ⚠ 只统一量纲，**不等于校准正确**——实证校准（从回归器残差估计 σ）是独立后续项目。
    # 经 SessionConfig.precision_commensurable → to_state_flags() → ainvoke 贯通。
    precision_commensurable: bool = False  # 默认关=零回归

    # ── 硬门摘出数值通路（议会第三轮 D1；ZERO_IGNITION_GATE_FUSION env）──
    # ⚠ **方向与其它旋钮相反**：默认 **True = 门关 = 逐字旧行为**（硬门同时决定「谁进
    #   fuse_terms」与「谁可报告」）。设 False 才是新架构：硬门只留报告通路，
    #   数值后验走**全流原生 (μ, Π) 精度加权**，不乘任何 gate/D 因子。
    # 神经席裁定：GNW ignition = 「什么内容变得**可报告**」，不是「谁计算数值」；
    #   **阈下不点燃 ≠ 阈下零影响**。同一开关一并解除 fast_survival_prior 的 arousal 地板
    #   （D5「失真必改」），杜绝「只修地板不改架构」这种未评审的中间态。
    # 仅 workspace 分支消费（非 workspace 两条分支本就不经 ignite）。
    gate_fusion: bool = True  # 默认 True=门关=零回归（注意方向）
    # ── physio 流排除出数值通路（议会 D7·跨仓承诺；默认 True=排除）──
    # 配套项目 Zero_MCP 用 WESAD 真被试验证其 EDA arousal 与唤醒**系统性反号**，明确请求
    # 「宁可继续门掉——『暂时不参与融合』优于『以反号参与』」。由我方单边可控。
    # 仅在 gate_fusion=False 时有意义（门关时 physio 本就受硬门管）。
    exclude_physio_fusion: bool = True

    # text_coping 独立标量流（议会 2026-07-16 B3；来源：PerceptionAgent 词典/回归产出）。
    # ── text_coping_enabled：B3 总门控（默认 False=零回归）──
    # False → AppraisalAgent B3 强制 text=None，只走分支1/3（纯 ctrl 路径），逐字旧行为；
    # True → 允许 text_coping_prior 参与 B3 融合（分支2/4）。
    # 经 SessionConfig.text_coping_enabled → to_state_flags() → ainvoke 贯通。
    text_coping_enabled: bool = False  # 默认关=零回归
    # ── fear_domain_enabled：WARN-3 fear 专属门（B1 BLOCK 前置·议会 2026-07-21·A1）──
    # False（默认）→ 任何路径**不得产出 fear 域激活**（两条泄漏路径均须覆盖·零回归）：
    #   路径一（流卫生）：perception._compute_text_coping 中 survival_narrative 域信号硬弃
    #     （不论方向·anger confrontational 路径完全不受此门）。
    #   路径二（单点完整）：emotion_lexicon.motivational_system coping<COPING_FEAR_THRESHOLD
    #     分支返回 rage 而非 fear（保守默认·非 fear 域激活）。
    # True → 两路径解除硬弃/回退，fear 域激活可经正常门控产生（须 env 显式开）。
    # 语义边界：fear 专属门·anger confrontational 路径完全不受此门（仅 survival 域关）。
    # 边界·表情层正交（议会 2026-07-21 B-facs-fear·PASS）：本门仅治标签/符号层两路径；
    #   表情层 fear-AU（affect_math.py:595-604·AU01/02/20·coping<0 驱动）由 facs_extended
    #   +coping_potential_enabled 双层容量门独立治理·与本门正交（面部运动 vs 情绪判定层解离·
    #   Rinn 1984 / Barrett 2019）。悬置 B-facs-fear-unlock：fear 域解锁时须重裁表情层是否加门。
    fear_domain_enabled: bool = False  # WARN-3 fear 专属门·默认关=零回归·B1 BLOCK 前置
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
    # ── recalled_episode_ids：本轮语义召回命中的 episode rowid 列表 ──
    # MemoryRecallAgent 填写；Supervisor 任务完成节点读出并经 MemoryClient
    # 节流更新 access_count（ACT-R 频率）。每轮 step() 归零防 LastValue 残留
    # （仿 external_priors 先例，见 runner.py）；不加 pydantic 约束防反序列化失败。
    recalled_episode_ids: list[str] = Field(default_factory=list)
