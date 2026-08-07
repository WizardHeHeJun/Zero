"""情感流水线的纯数学内核（可独立单测）。

贝叶斯流水线：OCC 先验 → TD/精度 → 高斯积融合 → 后验采样 → 通道解码。
含数值钳制：精度/方差下限、采样方差上界，保证不发散。
此模块为纯函数，无 I/O、无副作用。
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

# 运行期 import（不是仅注解）：`expand_external_priors` 真的要抛它。
# 同层导入（`project-root.md`：`src/agents/` 与 `src/orchestration/` 同属编排层），
# 且 external_prior 是零依赖叶子协议模块——不破坏本模块「纯函数、无 I/O、无副作用」。
from src.orchestration.external_prior import ExternalPriorError

if TYPE_CHECKING:
    # 仅注解用（PEP 563 惰性注解下不产生运行时 import）——external_prior 是编排层叶子协议
    # 类型（无任何 import），置于 TYPE_CHECKING 顶部消 code-reviewer W1 的末尾延迟 import 味道。
    from src.orchestration.external_prior import ExternalPrior

# 数值边界，避免精度/方差退化或采样发散
MIN_SIGMA = 0.05
MIN_PRECISION = 1e-3
MAX_SAMPLE_SIGMA = 0.5

# Mood（A.7）慢变心境双稳动力学参数：self_gain*self_k > 1-inertia 时双稳（pitchfork）
MOOD_INERTIA = 0.6
MOOD_SELF_GAIN = 0.5
MOOD_SELF_K = 2.0
MOOD_DRIVE = 0.2
MOOD_PRECISION = 0.8

# Language（affect↔language 双向收敛回路）
LANGUAGE_TOLERANCE = 0.15  # 语言情感与内核 e* 的一致阈值 τ
RECONCILE_WEIGHT = 0.5  # 双向互调：e* 向语言情感拉拢的权重

# Workspace（v3 显著度门控全局工作空间，默认关）：并行流竞争 → ignition 广播
SURVIVAL_PRECISION = 0.4  # 快生存流（上丘-枕-杏仁核捷径）：低固定精度（粗、快）
TEXT_AFFECT_PRECISION = 0.3  # 文本语义流精度（固定低值，显式低于 occ_prior 动态精度）
# 初版固定：类比 SURVIVAL_PRECISION 的工程近似（Friston 2005 允许初始固定精度）；
# 未来可从回归器残差动态估计（议会悬而未决 #1）。
# 行为反馈流精度（行为反馈环第二步·议会 2026-08-07 设计门）：显式低于 text（0.3）。
# ⚠ 此值是**跨通道类比得出的宽松上界，非精确估计**（生物席 #13）：证据载体是皮肤电/
# 面部肌电（表情反馈/SMH 文献），套到头部运动学是类比外推；头部/躯干粗大动作更易被
# 随意控制覆盖，信噪比不优于面部肌电。共线性折价框架**不适用**（C 主干已结构性剔除
# 与 mood 冗余的分量，议会必改 #5）。
BEHAVIOR_PRECISION = 0.15
# 行为流权重上界（议会必改 #2·数学席失真裁定的修法）：前置 cap π_b 约束不了 w_b
# （全弱流下 w_b→0.97；含 survival 地板的最坏情形 0.27）⇒ 融合前按实际 terms 现算
# w_b 并后置封顶（cap_stream_weight）。候选 0.15，终值待补维仿真确认（议会悬而未决）。
W_MAX_BEHAVIOR = 0.15

# ── 精度量纲齐次化（议会 2026-07-28 第四轮；PRP/精度量纲齐次化/）─────────────
# 问题：上面几个常量与 occ_prior 的 arousal_gain/σ² **不是同一物理量**。把它们当 1/σ²
# 反解得 σ_survival=1.58 / σ_mood=1.12 / σ_text=1.83——全部宽于 [-1,1] 值域的一半，
# 说得通的唯一方式是「这些流几乎什么都不知道」，那 Π 就该趋近 MIN_PRECISION。
# 它们从来不是按方差倒数设计的，只验证过**序**（ordinal, Stevens 1946），
# 而加权平均需要**比值尺度**（ratio-scale）。
# 本节把各流改写成同一把尺子（都是 1/σ²，σ 表达在 [-1,1] 值域上），
# **仅统一量纲、不等于校准正确**——实证校准（从回归器残差估计 σ）是独立后续项目。
# 全部经 state.precision_commensurable 门控，默认关 = 逐字旧行为。
SURVIVAL_CONF_BASE = 0.1  # 快生存流置信度截距（远低于 occ_prior 的 0.3）
SURVIVAL_CONF_GAIN = 0.1  # 置信度随 |intensity| 的斜率（远低于 occ_prior 的 0.5）
SURVIVAL_CONF_CAP = 0.5  # 置信度上限：保证 σ_survival ≥ 0.25，天然弱于 appraisal
# σ_mood = 1.0：mood_step 的双稳吸引子实测落在 clamp 边界 ±1（非内部不动点——
# f(m)=inertia·m+self_gain·tanh(self_k·m)−m 在 (0,1] 上恒 >0，一路推到边界），
# 不稳定不动点在 0 → **盆半宽 = 1.0**。心境作为「当前瞬时情感」的先验，
# 其不确定性就是它能漂多远，故 σ_mood 取盆半宽。（现常数 0.8 反解 σ=1.118，量级吻合。）
SIGMA_MOOD = 1.0
# σ_text = 0.5：代码原注释自陈「显式低于 occ_prior 动态精度」，occ_prior 的 σ∈[0.10,0.35]，
# 故 σ_text 须 >0.35；又应强于 mood（针对当前输入而非跨轮基调），故 <1.0。
# 取 0.5（≈ occ 最宽 0.35 与 mood 1.0 的中段）。**保守占位，待实证校准**。
SIGMA_TEXT = 0.5
# σ_bhv = 1.3：行为反馈流齐次档 σ（初值占位·由 scripts/calibrate_sigma_bhv.py 回填）。
# 判据（议会必改 #4 的三处统计口径，校准脚本落实）：① 取回合间 |Δa| 分布的 95 分位**本身**
# 作保守上界（高估 σ = 低估精度，方向安全；分位数 ≈1.96σ，非 σ 的无偏估计——刻意为之）；
# ② 开环统计校准闭环参数属自指校准 ⇒ σ 定值后须闭环重测 |Δa| 一致性检验；
# ③ |Δa| 只覆盖「漂移」分量，「调节偏移」|a_expr−a_felt| 是回合内量、单独测量分开报告。
# 须 ≥ σ_mood（1.0）：行为证据是表达侧代理，不确定性不低于心境基调。
SIGMA_BHV = 1.3
# value 流：precision_da 的 sigmoid(α|δ|)∈[0.5,1) 是概率不是逆方差，把它当 Π 反解得
# σ∈[1.00,1.41]（宽于整个值域）。重标定到 [MIN_PRECISION, VALUE_PRECISION_CEILING]，
# 上限取与 survival 同量级（同属快速粗略通路），使二者可比。
VALUE_PRECISION_CEILING = 6.25
SALIENCE_THRESHOLD = 0.18  # ignition 阈值：salience 低于此的流不点燃（停留局部）
AROUSAL_GAIN = 1.0  # NE/唤醒对评价·价值流精度的增益系数（唤醒越高投票权越大）
# ignite 软门控陡度（议会 2026-07-02 Item 2）：None=硬 step 零回归（默认）；
# 非 None → logistic gate(sᵢ;θ,β)=σ(β·(sᵢ−θ)) 软化 GNW all-or-none 近似。
# 推荐区间 [20,50]（神经/数学席交集），典型值 20；β<1 无意义（gate 趋 0.5 均匀融合）。
# 由 ZERO_IGNITION_BETA env（默认注释关）→ state.ignition_beta → ignite(soft_beta=...) 注入。
IGNITION_BETA: float | None = None
LANG_BASE_PRECISION = 1.0  # 精度加权再入里语言侧的基准精度（与内核后验精度竞争）

# 情绪时间尺度分层（affective chronometry + ALMA/WASABI）：快变情绪向「态度基线」衰退、被刺激冲击；
# 慢变态度对「对象/人」长期累积（evaluative conditioning）。情绪是短时的——不长期累积。
EMOTION_RECOVERY = 0.4  # 情绪向基线的残留比例（小=恢复快；过大=emotional inertia 病理）
EMOTION_REACTIVITY = 0.6  # 对当前刺激的即时反应增益
ATTITUDE_RATE = 0.08  # 态度对刺激的慢累积率（多轮才成形；越小越稳）
# 态度向「个体习惯性基线 setpoint」的弱均值回归（议会 B-1 必改：缺此项则持续同向刺激下 attitude
# 单调漂移到极端=affective homeostasis 缺失 + emotional inertia 病理 / 慢性应激无 HPA 负反馈）。
# 量级远小于 ATTITUDE_RATE：可被条件化、但不无限累积。稳态 a*≈rate·s/(rate+reversion)（<|s|）。
# 心理席建议 0.005–0.01、生物席 0.01–0.02 → 取交叠处 0.01。setpoint 默认中性，将来人格阶段由
# 大五（Agreeableness/Neuroticism）PAD 偏置先验替换。文献见议会纪要 / Russell 2003 · Kuppens 2010。
ATTITUDE_REVERSION = 0.01  # 态度向 setpoint 的每轮回归率（0=关，退化为纯 EWMA 累积=旧行为）
ATTITUDE_SETPOINT = (0.0, 0.0)  # 个体习惯性情感基线（无偏人格；attitude/emotion 回归的锚）

# Regulation 重评（Gross 过程模型）：reappraisal 改「构念/意义」而非末端压制
REAPPRAISAL_ANCHOR = 0.1  # 重评把负/低效价重新解释、向其拉拢的「积极锚」
REAPPRAISAL_LIFT = 0.7  # 向积极锚拉拢的比例（重评改变体验，不只是表达）
REAPPRAISAL_CALM = 0.4  # 重评对唤醒的平复系数（威胁被重构 → 唤醒下降）

# HPA/皮质醇慢回路（P3 1-B；推荐值；实际运行经 env 注入，代码常量仅为初始化参考）
# 生物席（Herman 2016；Becker & Rohleder 2019；PMC8139339）：血浆半衰期 60-70min → τ≈86-100min≈5400s
CORTISOL_TAU_DECAY: float = 5400.0  # 衰减时间常数（秒）；~90min 对应血浆半衰期/ln2
CORTISOL_IMPULSE: float = 0.7  # 单次应激脉冲注入量（归一 [0,1]；0.6-0.8 生物席推荐）
CORTISOL_CAP: float = 1.0  # 皮质醇上界（归一；对应肾上腺分泌生理上限）；防 runaway 兜底
CORTISOL_THETA_GOAL: float = 0.3  # 触发判据：目标不一致性阈值（Dickerson & Kemeny 2004）
CORTISOL_THETA_INTENSITY: float = 0.5  # 触发判据：强度阈值（同引用；不可控性+高强度=HPA激活）


def sigmoid(x: float) -> float:
    """数值稳定的 logistic 函数。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    """把 x 钳制到 [lo, hi]。"""
    return max(lo, min(hi, x))


def occ_prior(
    goal_congruence: float,
    standard_compliance: float,
    attitude_appeal: float,
    intensity: float,
    *,
    arousal_baseline: float = 0.0,
    va_coupling_pos: float = 0.6,
    va_coupling_neg: float = 0.6,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """OCC 评价 → (prior_mu, prior_sigma, reward)。

    valence 由目标/标准/态度一致性线性合成；arousal 由强度与 |效价| 合成；
    sigma 随显著度上升而下降（越显著越确定）；reward = 目标一致性（闭合 2↔3）。
    `arousal_baseline`（默认 0.0，零回归）平移 arousal 证据基准，设负值启用 deactivation（Q7）。
    A-P2-A：`va_coupling_pos/neg`（均默认 0.6，零回归）拆自原 `0.6·|valence|`，允许负效价侧
    使用更高系数（Kuppens 2013 negativity bias；推荐 neg=0.7/pos=0.5，但不改默认，走配置注入）。
    """
    valence = clamp(
        0.5 * goal_congruence + 0.3 * standard_compliance + 0.2 * attitude_appeal,
        -1.0,
        1.0,
    )
    # P1-c（议会 Q7·失真必改）：`arousal_baseline` 让平淡输入可给零/负 arousal 证据（副交感
    # deactivation 臂；circumplex 下半区）。默认 0.0 = 逐字旧行为（arousal 恒正整流，零回归）；
    # 设负值（如 -0.08 抵消 0.4·下限）则低强度低 |valence| 的平淡对话把 arousal 拉向静息/负。
    # A-P2-A va_coupling 非对称（Kuppens 2013 negativity bias：负效价侧 arousal 斜率更陡）：
    # `va_coupling_neg*max(-valence,0) + va_coupling_pos*max(valence,0)` 拆自原 `0.6*|valence|`。
    # 默认 0.6/0.6 = 逐字旧行为（零回归）；推荐 neg>pos（如 0.7/0.5）但不改默认，走后续配置。
    arousal = clamp(
        0.4 * abs(intensity)
        + va_coupling_neg * max(-valence, 0.0)
        + va_coupling_pos * max(valence, 0.0)
        + arousal_baseline,
        -1.0,
        1.0,
    )
    conf = clamp(0.3 + 0.5 * abs(intensity), 0.0, 1.0)
    sigma = max(MIN_SIGMA, 0.5 * (1.0 - conf))
    reward = clamp(goal_congruence, -1.0, 1.0)
    return (valence, arousal), (sigma, sigma), reward


def td_update(
    value_table: dict[str, float],
    key: str,
    reward: float,
    *,
    gamma: float = 0.9,
    lr: float = 0.2,
    next_value: float = 0.0,
) -> tuple[float, float, dict[str, float]]:
    """在线 TD 更新 V(s)，返回 (delta, new_value, updated_table)。

    delta = reward + gamma * next_value - V(s)；V(s) += lr * delta。
    返回新表（不原地 mutate 入参）。
    """
    v_s = value_table.get(key, 0.0)
    delta = reward + gamma * next_value - v_s
    new_v = v_s + lr * delta
    updated = dict(value_table)
    updated[key] = new_v
    return delta, new_v, updated


def precision(
    delta: float,
    value_estimate: float,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    commensurable: bool = False,
) -> float:
    """精度 π = σ(α·|δ| + β·V)：RPE 强度与价值确定性共同决定证据权重。

    `commensurable=True`（量纲齐次化，默认关）：与 `precision_da` 同一处理——sigmoid 输出
    是概率不是逆方差，重标定到 [MIN_PRECISION, VALUE_PRECISION_CEILING]。
    ⚠ 本函数是**默认路径**（`affect_core.py` 的 `gaussian_fuse` 分支）的证据精度来源
    （`value.py` 的 `precision()` 调用 → `state.precision`），
    故此门是三条融合分支里唯一每轮无条件生效的一处。
    """
    if commensurable:
        return MIN_PRECISION + (VALUE_PRECISION_CEILING - MIN_PRECISION) * sigmoid(
            alpha * abs(delta) + beta * value_estimate
        )
    return max(MIN_PRECISION, sigmoid(alpha * abs(delta) + beta * value_estimate))


def precision_da(delta: float, *, alpha: float = 1.0, commensurable: bool = False) -> float:
    """DA 路径精度 π_DA = σ(α·|δ|)：消去 β·V，仅由 RPE 幅度决定证据权重。

    议会裁决 A-P1-A（神经席 M5 + 数学席 M6）：`precision(δ, V)` 原式 σ(α|δ|+β·V) 把
    DA 精度与价值混同，β·V 无神经依据。此函数只保留 DA 通路真正编码的信号——预测误差
    幅度 |δ|；value_estimate 项移除。精度下界与 `precision` 保持一致（MIN_PRECISION 钳制）。
    纯函数、无 I/O、无 env（守热路径红线）。

    `commensurable=True`（第四轮议会量纲齐次化，默认关）：`sigmoid()` 输出是 **概率**
    不是 **逆方差**，直接当 Π 送进 `fuse_terms` 会反解出 σ∈[1.00,1.41]——宽于整个
    [-1,1] 值域，语义上等于「这条流什么都不知道」，与它实际占的权重矛盾。
    重标定到 [MIN_PRECISION, VALUE_PRECISION_CEILING]，保持单调性与「RPE 越大权重越高」
    的原语义不变，只换尺度。
    """
    if commensurable:
        return MIN_PRECISION + (VALUE_PRECISION_CEILING - MIN_PRECISION) * sigmoid(
            alpha * abs(delta)
        )
    return max(MIN_PRECISION, sigmoid(alpha * abs(delta)))


def effective_stream_count(
    terms: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float]:
    """Kish 有效样本量 `N_eff = (ΣΠ)²/ΣΠ²`，逐维返回。**纯观测量，不参与任何计算。**

    读数：`N_eff → 1` = 后验实际上只由一条流决定（其余流的权重被碾压）；
    `N_eff → N` = N 条流均衡贡献。齐次化前实测均值 1.175（几乎恒为单流独裁），
    齐次化后 2.117。

    ⚠ **不得据此下硬断言**——`N_eff → 1` 在**合法**校准下也会发生（Ernst & Banks 2002：
    某模态噪声趋零时它的权重本就该趋 1）。它回答的是「有几条流在说话」，
    不回答「这样对不对」。写进 trace 供观测/排障，不做门。

    Kish, L. (1965). *Survey Sampling*. Wiley. —— 加权估计量的有效样本量定义。
    """
    out = []
    for d in (0, 1):
        ps = [max(MIN_PRECISION, prec[d]) for _, prec in terms]
        s1 = sum(ps)
        s2 = sum(p * p for p in ps)
        out.append(s1 * s1 / s2 if s2 > 0.0 else 0.0)
    return (out[0], out[1])


def mood_precision(*, commensurable: bool = False) -> float:
    """心境流精度。门开时 = 1/σ_mood²（σ_mood = mood_step 吸引盆半宽，见 SIGMA_MOOD）。"""
    return 1.0 / SIGMA_MOOD**2 if commensurable else MOOD_PRECISION


def text_affect_precision(*, commensurable: bool = False) -> float:
    """文本语义流精度。门开时 = 1/σ_text²（σ_text 见 SIGMA_TEXT，保守占位待实证校准）。"""
    return 1.0 / SIGMA_TEXT**2 if commensurable else TEXT_AFFECT_PRECISION


def behavior_precision(*, commensurable: bool = False) -> float:
    """行为反馈流精度。门开时 = 1/σ_bhv²（σ_bhv 见 SIGMA_BHV，占位待校准脚本回填）。"""
    return 1.0 / SIGMA_BHV**2 if commensurable else BEHAVIOR_PRECISION


def behavior_feedback_evidence(
    copy: dict[str, Any],
    *,
    commensurable: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """把上一回合的 motion_efference 副本映射成行为反馈流证据 (μ, Π)；缺席返回 None。

    行为反馈环第二步（议会设计门 2026-08-07 收敛裁决：C 主干 + regulation_strategy
    分流解释层，纪要 notes/2026-08-07-behavior-feedback-council.md）：

    - **在场门 = 副本的 voluntary 路非 None**（第一步定死的回退语义：voluntary 为 None
      当且仅当未开调节或调节没改变什么 ⇔ 调节差 δ≡0）。δ≡0 ⇒ 返回 None = 流缺席
      （absent-cue，不注入 0 值假证据）。生产默认 regulation 关 ⇒ 流恒缺席 = 零回归。
    - **μ = (0, a_expr)**，其中 a_expr = 2·voluntary.onset − 1（onset=(a+1)/2 的精确仿射逆，
      无增益依赖）——「表达出来的唤醒水平」。⚠ 位置空间口径（design.md §1.2）：主干信号
      δ = a_expr − a_felt 是位移量，直接当 μ 喂是范畴错误（会把「压低了 0.4」误宣称为
      「唤醒是 −0.4」）；条件在场 + 位置空间 a_expr 的边际信息量恰为 δ。
    - **Π = (MIN_PRECISION, π_b)**：valence 维置底（动作层无 valence 信息，议会既有裁定）；
      arousal 维 π_b 见 behavior_precision 两档。

    语义标注（议会必改 #9·心理×神经独立收敛的失真修正）：suppression 下 δ<0 的语义是
    **「表达抑制幅度」**（外显行为构造量）——🛑 不得表述为「压制使体验变平静」：
    Gross & Levenson 1993 / Webb 2012 实证压制**不改主观体验、交感反增**，`regulation`
    模块注释的表达/体验区分是正确基准。reappraisal 下 δ 反映重评后的真实变化，且 mood
    只吃调节前 e*（MoodAgent 在 regulation 之前），δ 是 mood 不含的独立信息。
    数字人表达=恒被观察，接近 Noah 2018「效应消失」边界条件——低增益/流缺席默认的
    额外支持（必改 #10）。

    机制归类与时标（议会必改 #11/#12）：本操作借用的是 Wolpert-Ghahramani-Jordan 1995
    「由自身运动指令副本反推隐藏状态」的抽象拓扑，**不是** Crapse & Sommer 2008 经典
    corollary discharge（同模态感觉抵消）；骨骼肌系统自有 corollary discharge/小脑前向
    模型/γ-fusimotor 通路（比内脏运动类比更贴），但「运动学状态→情绪状态」这一跳仅有
    Seth 2013 一般 active inference 框架兜底，是**工程外推非直接证据**。时标免责：本反馈
    是回合级（秒~十秒、延迟受用户交互节奏支配、无固定时间常数），与生物 corollary
    discharge 毫秒级前馈不同量级，仅借拓扑不主张时标对应。

    🛑 单射性前提（议会必改 #14·未来强制项）：仿射逆的精确性绑定在
    `motion_synth.modulation_from_affect` 为解析回退（仿射单射）这一实现上；将来真模型
    替换该函数时，本映射不得沿用——须先校验单射性或改为直传 arousal 标量，否则
    「反推忠实」判定失效。

    Args:
        copy: AffectState.motion_efference（恰好上一回合的指令级副本；staleness 修正后
            该字段恒为上一回合产出或 None，调用方先判 None 再进本函数）。
        commensurable: 精度量纲齐次化门（state.precision_commensurable）。

    Returns:
        (μ, Π) 或 None（流缺席）。scene=="speaking" 时 raise NotImplementedError——
        speaking 场景预留结构（PRD D3），TTS 韵律接入前无写入者，显式阻塞而非空壳。
    """
    if copy.get("scene") == "speaking":
        raise NotImplementedError(
            "behavior_feedback_evidence 的 speaking 分支未定义：speaking 场景的副本"
            "须待 TTS 韵律流接入后由议会补裁（PRD D3 预留结构，显式阻塞非遗漏）"
        )
    voluntary = copy.get("voluntary")
    if voluntary is None:
        return None  # δ≡0 ⇒ 流缺席（absent-cue），不给 0 值假证据
    a_expr = clamp(2.0 * float(voluntary["onset"]) - 1.0, -1.0, 1.0)
    pi_b = behavior_precision(commensurable=commensurable)
    return ((0.0, a_expr), (MIN_PRECISION, pi_b))


def cap_stream_weight(
    terms: list[tuple[tuple[float, float], tuple[float, float]]],
    names: list[str],
    *,
    target: str,
    w_max: float = W_MAX_BEHAVIOR,
    dim: int = 1,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str]]:
    """对实际参与融合的 terms 后置封顶 target 流在 dim 维的权重 w = π/Σπ ≤ w_max。

    议会必改 #2（数学席失真裁定的修法）：前置 cap π_b 是对上游常数的假设性推演，
    约束不了凸组合里的实际权重——其余流全弱（各 MIN_PRECISION）时 w_b→0.97，即便算上
    survival 的 arousal 地板（gate_fusion 副作用、非正式不变量）最坏仍 0.27。本函数在
    ignite 之后、fuse_terms 之前对**本轮真正进入融合的流集合**做不变量校验，超界则
    重标定 π_target′ = w_max/(1−w_max)·Σ_{j≠target}π_j，使封顶后权重恰为 w_max。

    ⚠ 地板边界（实现期变异测试抓出的议会公式缺陷，须随必改 #2 交数学席复核）：
    fuse_terms 对每项精度取 max(MIN_PRECISION, ·)——当重标定值 π′ < MIN_PRECISION
    （其余流全部触底的退化情形），地板会把 π′ 抬回去、封顶静默失效（n 条全底流时
    目标流最低只能拿到 1/(n+1) 均分票）。地板之下目标权重**数学上不可达** ⇒ 该情形
    改为**整条流剔除**（absent 语义：全沉默环境里 lag-1 行为回声不该获得均分投票权，
    与「不给 0 值假证据」同一哲学）。terms 与 names **成对返回**保持对齐
    （BLOCK 1 先例：两者须同一次筛选产出，防 zip 失配）。

    target 不在 names 中或未超界 ⇒ 原 (terms, names) 原样返回；不 mutate 输入。
    纯函数，供 affect_core 融合调用点与稳定性仿真共用。
    """
    if len(terms) != len(names):
        raise ValueError(f"terms 与 names 长度不一致：{len(terms)} != {len(names)}")
    try:
        idx = names.index(target)
    except ValueError:
        return terms, names
    total_other = sum(
        max(MIN_PRECISION, prec[dim]) for j, (_, prec) in enumerate(terms) if j != idx
    )
    pi_target = max(MIN_PRECISION, terms[idx][1][dim])
    capped_pi = w_max / (1.0 - w_max) * total_other
    if pi_target <= capped_pi:
        return terms, names
    if capped_pi < MIN_PRECISION and len(terms) > 1:
        # 地板之下封顶不可达 ⇒ 剔除目标流（见 docstring ⚠）。len==1 时不剔（fuse_terms
        # 空输入 raise，且单流场景 w=1 无从封顶——留给调用方的流装配保证不出现）。
        pruned_terms = [t for j, t in enumerate(terms) if j != idx]
        pruned_names = [n for j, n in enumerate(names) if j != idx]
        return pruned_terms, pruned_names
    mu, prec = terms[idx]
    new_prec = (prec[0], capped_pi) if dim == 1 else (capped_pi, prec[1])
    out = list(terms)
    out[idx] = (mu, new_prec)
    return out, names


def evidence_from_value(reward: float, delta: float) -> tuple[float, float]:
    """把 reward/RPE 映射成 valence-arousal 证据均值。

    reward → valence 位移；|delta|（意外度）→ arousal 位移。
    """
    return (clamp(reward, -1.0, 1.0), clamp(abs(delta), -1.0, 1.0))


def gaussian_fuse(
    prior_mu: tuple[float, float],
    prior_sigma: tuple[float, float],
    evidence_mu: tuple[float, float],
    pi: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """先验与证据的高斯积融合（按精度加权），逐维返回 (post_mu, post_sigma)。

    prior 精度 = 1/sigma^2；证据精度 = pi。
    """
    post_mu: list[float] = []
    post_sigma: list[float] = []
    for p_mu, p_sig, e_mu in zip(prior_mu, prior_sigma, evidence_mu, strict=True):
        prior_prec = 1.0 / max(MIN_SIGMA, p_sig) ** 2
        post_prec = prior_prec + pi
        mu = (prior_prec * p_mu + pi * e_mu) / post_prec
        sig = math.sqrt(1.0 / post_prec)
        post_mu.append(clamp(mu, -1.0, 1.0))
        post_sigma.append(clamp(sig, MIN_SIGMA, MAX_SAMPLE_SIGMA))
    return (post_mu[0], post_mu[1]), (post_sigma[0], post_sigma[1])


def fuse_terms(
    terms: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """多项高斯先验/证据按精度融合，逐维返回 (post_mu, post_sigma)。

    每项 = (mu, precision)（逐维精度）；后验精度 = 各项精度之和。
    仅含「先验 + 证据」两项时与 gaussian_fuse 数值一致（见 test）。

    空输入 → `ValueError`（design D12）。此前是 `den=0` 一路走到 `num/den` 抛
    **无信息的 `ZeroDivisionError`**；本函数是全仓融合的公共入口，值得一句指名道姓的报错。
    **非空输入逐位不变**——纯错误信息改善，非行为变更。
    """
    if not terms:
        raise ValueError(
            "fuse_terms 收到空的 terms：至少需要一项 (μ, Π) 才能定义后验。"
            "调用方应保证流列表非空（如 ignite 的 fusion_terms 前置条件）"
        )
    post_mu: list[float] = []
    post_sigma: list[float] = []
    for d in range(2):
        num = 0.0
        den = 0.0
        for mu, prec in terms:
            p = max(MIN_PRECISION, prec[d])
            num += p * mu[d]
            den += p
        post_mu.append(clamp(num / den, -1.0, 1.0))
        post_sigma.append(clamp(math.sqrt(1.0 / den), MIN_SIGMA, MAX_SAMPLE_SIGMA))
    return (post_mu[0], post_mu[1]), (post_sigma[0], post_sigma[1])


def hierarchical_fuse(
    named_terms: list[tuple[str, tuple[float, float], tuple[float, float]]],
    *,
    layers: int = 1,
    coupling: float = 0.0,
    low_names: frozenset[str] = frozenset({"survival", "mood"}),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """层级预测编码融合（v1 = 2 层，设计门定稿：design.md 六-bis / 数学席二轮）。

    架构：
      L0（感觉-唤醒层）= name ∈ low_names 的项（survival + mood 粗 VA，精度较低、时标快）；
      L1（核心情感层）= 其余项（appraisal / value / text 竞争层）。
    输出 e* = L1 核心情感层后验（Barrett 2017 前岛叶 / ACC 整合预测，输出取整合层非最低层）。

    五步闭式（w = coupling，逐维 d∈{0,1} 独立，Friston 2008 精度加权预测误差调制）：
      ① L0 融合：   π_L0  = Σ_{i∈L0} π_i ;  μ_L0  = Σ_{i∈L0} π_i·μ_i / π_L0
      ② L1 证据：   π_L1e = Σ_{j∈L1} π_j ;  μ_L1e = Σ_{j∈L1} π_j·μ_j / π_L1e
      ③ 误差：      ε = μ_L0 − w·μ_L1e            （bottom-up，供观测；不单独入后验）
      ④ L0→L1 误差项：均值 = μ_L0，精度 = w²·π_L0  （平方耦合精度调制）
      ⑤ 核心后验：  π_core = π_L1e + w²·π_L0
                    μ_core = (π_L1e·μ_L1e + w²·π_L0·μ_L0) / π_core
                    σ_core = sqrt(1/π_core) → clamp(MIN_SIGMA, MAX_SAMPLE_SIGMA)
                    μ_core → clamp(-1, 1)

    双退化（逐字等价 fuse_terms(all_terms)，<1e-9 零浮点误差）：
      - layers == 1 → 无层间递推，直接 fuse_terms（平层零回归）
      - coupling == 0.0 → 关层级，退平层（L1 排除 survival+mood 则不等价 fuse_terms(all)；
        定义 coupling=0 = 关层级，走同一 fuse_terms 代码路径，零浮点误差）

    边界：
      - L0 空 → fallback fuse_terms(all)（等价 layers=1）
      - L1 空 → fallback fuse_terms(all)（保分母非零）
      - coupling > 1.0 → raise ValueError（硬拒不 clamp；>1 破坏凸组合 → 类 seeking 单侧锁定）
      - layers > 2 → raise ValueError（v1 只实现两层；≥3 会与 2 同结果，硬拒不静默降级）

    有界性/稳定性（数学席证）：
      μ_core 是 μ_L1e 与 μ_L0 的精度加权凸组合（w≤1 → w²·π_L0/π_core≤1），
      ∈[-1,1] 天然成立（仍保留 clamp 防御）；谱半径 ρ≤1，无发散不动点。

    coupling 约束（CS 红线）：
      coupling 是 env 注入的**固定浮点标量**，非情感状态 (v,a) 的函数。
      **禁止可学习 / 动态 / 运行期自适应 / 状态相关耦合**（否则非线性 feedback 破稳定性）。
    """
    # ── 退化旁路（最先判，两条均走 fuse_terms 同一代码路径，<1e-9 零浮点误差）──
    all_terms = [(mu, prec) for _, mu, prec in named_terms]
    if layers == 1 or coupling == 0.0:
        return fuse_terms(all_terms)

    # ── 入口校验（硬拒不 clamp；coupling==0.0 已被上方退化旁路吃掉，此处 coupling>0）──
    # 🛑 `layers` 必须在这里校验，理由与 coupling 同：v1 **只实现两层**，本函数从不做
    # 层数递推 —— `layers` 全程只被上方 `layers == 1` 用过一次。不校验的话 layers=3/7/10**9
    # 与 layers=2 **逐字同结果**（2026-07-30 实测），而 `.env.example` 的 ZERO_HPC_LAYERS
    # 明写「层数 1-2」⇒ 配置面收下一个引擎不兑现的值、两侧都不报错，消费方以为开了 4 层
    # 预测编码。这与 R11 那族「回报值 ≠ 生效语义」同型，故同样硬拒、不静默降级。
    # 位置刻意在退化旁路**之后**：layers≥3 但 coupling==0.0 = HPC 明确关闭，
    # 不为「没启用的旋钮的值」抛错（与 coupling>1 在 layers==1 时不抛错一致）。
    if layers > 2:
        raise ValueError(
            f"layers={layers} 超出已实现层数（v1 = 2 层：L0 感觉-唤醒 / L1 核心情感）；"
            f"本函数不做层数递推，≥3 会与 layers=2 同结果而非真的多层 —— 故硬拒不静默降级。"
            f"请将 ZERO_HPC_LAYERS 设为 1（平层）或 2（启用 HPC）。"
        )
    if coupling < 0.0 or coupling > 1.0:
        raise ValueError(
            f"coupling={coupling} 超出语义范围 [0, 1]（design w∈(0,1]）；"
            f"负值=反向 top-down 语义不明；>1.0 破坏凸组合稳定性（谱半径 >1，类 seeking 锁定）；"
            f"请将 ZERO_HPC_COUPLING 设为 [0, 1]（推荐 [0.3, 0.8]）。"
        )

    # ── 分桶：L0（感觉层）/ L1（核心情感层）──
    l0: list[tuple[tuple[float, float], tuple[float, float]]] = []
    l1: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for name, mu, prec in named_terms:
        if name in low_names:
            l0.append((mu, prec))
        else:
            l1.append((mu, prec))

    # ── 边界：L0 空 / L1 空 → fallback fuse_terms(all)（保分母非零）──
    if not l0 or not l1:
        return fuse_terms(all_terms)

    # ── 五步闭式（逐维 d∈{0,1} 独立）──
    w = coupling
    w2 = w * w

    post_mu: list[float] = []
    post_sigma: list[float] = []
    for d in range(2):
        # ① L0 融合
        num_l0 = 0.0
        den_l0 = 0.0
        for mu_item, prec_item in l0:
            p = max(MIN_PRECISION, prec_item[d])
            num_l0 += p * mu_item[d]
            den_l0 += p
        pi_l0 = max(MIN_PRECISION, den_l0)
        mu_l0 = num_l0 / pi_l0

        # ② L1 证据融合
        num_l1 = 0.0
        den_l1 = 0.0
        for mu_item, prec_item in l1:
            p = max(MIN_PRECISION, prec_item[d])
            num_l1 += p * mu_item[d]
            den_l1 += p
        pi_l1e = max(MIN_PRECISION, den_l1)
        mu_l1e = num_l1 / pi_l1e

        # ⑤ 核心后验（④ L0→L1 误差项精度 = w²·π_L0，均值 = μ_L0；直接合并入 ⑤）
        pi_core = pi_l1e + w2 * pi_l0
        pi_core = max(MIN_PRECISION, pi_core)
        mu_core_raw = (pi_l1e * mu_l1e + w2 * pi_l0 * mu_l0) / pi_core
        post_mu.append(clamp(mu_core_raw, -1.0, 1.0))
        sigma_core = math.sqrt(1.0 / pi_core)
        post_sigma.append(clamp(sigma_core, MIN_SIGMA, MAX_SAMPLE_SIGMA))

    return (post_mu[0], post_mu[1]), (post_sigma[0], post_sigma[1])


def mood_step(
    prev_mood: tuple[float, float],
    affect: tuple[float, float],
    *,
    inertia: float = MOOD_INERTIA,
    self_gain: float = MOOD_SELF_GAIN,
    self_k: float = MOOD_SELF_K,
    drive: float = MOOD_DRIVE,
) -> tuple[float, float]:
    """慢变心境的双稳松弛：m' = inertia·m + self_gain·tanh(self_k·m) + drive·affect。

    self_gain·self_k > 1-inertia 时 pitchfork 双稳：正/负心境两个吸引盆。
    持续负向 affect 把 m 推入负盆后，轻微正向 affect 难以拉出 → 历史依赖/滞后。
    逐维钳制 [-1, 1]。纯函数、无副作用。
    """
    out: list[float] = []
    for m, a in zip(prev_mood, affect, strict=True):
        out.append(clamp(inertia * m + self_gain * math.tanh(self_k * m) + drive * a, -1.0, 1.0))
    return (out[0], out[1])


def affect_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """valence-arousal 平面上两情感点的欧氏距离（一致性度量）。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconcile_affect(
    e_star: tuple[float, float],
    lang_affect: tuple[float, float],
    *,
    weight: float = RECONCILE_WEIGHT,
) -> tuple[float, float]:
    """双向互调：把内核 e* 向语言反推情感按 weight 拉拢，逐维钳制 [-1, 1]。

    weight=0 保持 e*，weight=1 取 lang_affect，0.5 取中点。语言与情感相互判断时
    用此把 e* 朝语言侧微调（与「重写语言」一起构成双向收敛）。纯函数、无副作用。
    """
    v = clamp(e_star[0] + weight * (lang_affect[0] - e_star[0]), -1.0, 1.0)
    a = clamp(e_star[1] + weight * (lang_affect[1] - e_star[1]), -1.0, 1.0)
    return (v, a)


def reappraise(
    affect: tuple[float, float],
    *,
    anchor: float = REAPPRAISAL_ANCHOR,
    lift: float = REAPPRAISAL_LIFT,
    calm: float = REAPPRAISAL_CALM,
) -> tuple[float, float]:
    """认知重评（Gross 过程模型，早期干预）：改变对情境的「构念/意义」。

    与「表达抑制」（晚期、仅按比例压制输出幅度）不同，重评把负/低效价重新解释、
    向积极锚 `anchor` 拉拢（valence 上抬），并显著平复唤醒（威胁被重构 → 体验也变）；
    效价已高于锚则不再上抬（不无端拔高积极情绪）。逐维钳制 [-1, 1]，纯函数。
    """
    v, a = affect
    v2 = v + lift * (anchor - v) if v < anchor else v
    return clamp(v2, -1.0, 1.0), clamp(calm * a, -1.0, 1.0)


def sample_affect(
    post_mu: tuple[float, float],
    post_sigma: tuple[float, float],
    *,
    rng: random.Random | None = None,
    sigma_cap: float | None = None,
) -> tuple[float, float]:
    """从后验高斯采样 e*=(valence, arousal)；方差有界、结果钳制到 [-1, 1]。

    `sigma_cap` 为采样标准差上限（逐维），默认 None → 用常量 `MAX_SAMPLE_SIGMA`（逐字旧行为）。
    调小可让样本更贴 post_mu、降低逐轮 (v,a) 抖动（情绪「防抖」旋钮，经 state 由 env 注入）。
    仅钳采样方差，不动 `gaussian_fuse`/`fuse_terms` 里的后验数值稳定钳（另一回事）。纯函数、无 I/O。
    """
    cap = MAX_SAMPLE_SIGMA if sigma_cap is None else sigma_cap
    generator = rng if rng is not None else random.Random()
    v = clamp(generator.gauss(post_mu[0], min(cap, post_sigma[0])), -1.0, 1.0)
    a = clamp(generator.gauss(post_mu[1], min(cap, post_sigma[1])), -1.0, 1.0)
    return (v, a)


def text_label(valence: float, arousal: float) -> str:
    """按 valence-arousal 象限映射离散情绪词（占位与模型路径共用）。"""
    if valence >= 0:
        return "excited" if arousal >= 0.33 else "content"
    return "angry" if arousal >= 0.33 else "sad"


def decode_channels(
    affect: tuple[float, float],
    *,
    coping_potential: float = 0.0,
    facs_extended: bool = False,
    canonical_physiology: bool = False,
    k_arousal: float = 1.5,
    k_coping: float = 1.2,
) -> dict[str, Any]:
    """把 (valence, arousal) 占位解码为 4 个表达通道的结构化结果。

    通道：FACS AU 向量 / 文本情绪标签 / 生理信号模拟 / 语音韵律参数。

    Args:
        affect: (valence, arousal) 后验情感坐标，各维 ∈ [-1, 1]。
        coping_potential: 控制评价（AppraisalAgent 产出的 state.coping_potential_state），
            ∈ [-1, 1]。仅在 facs_extended=True 时生效；默认 0.0 无影响，零回归。
            正值→高控制感（愤怒方向），负值→低控制感（恐惧方向）。
            来源：直接读 state.coping_potential_state，**不在表情层重估**（防构念重复）。
        facs_extended: False（默认）→ 旧 5-AU 逐字行为，零回归；
            True → 输出 13-AU 扩展集合（含 coping 驱动的区分性 AU + AU17/AU26 通用 AU）。
        canonical_physiology: False（默认）→ legacy 占位 {hr[70,110]/sc[0,1]/pupil_mm[3,5]}，
            零回归；True → canonical 占位，议会 2026-07-23 公式：
            heart_rate_bpm=50+70·clamp(0.5·(1+arousal)) / skin_conductance=20·clamp(|arousal|) /
            temperature_c=36−3·clamp(|arousal|)（无 pupil_mm）。
            量纲与真 PhysiologyDecoder.predict_physiology 同口径（同域同方向）。
            注意：sc 中立态占位给 0μS，真 decoder 中立态 sigmoid≈0.5→~10μS，
            系统性偏低 ~10μS——禁止跨路径绝对比较（仅同路径内相对比较有意义）。
        k_arousal: ⚖ AU05/07 对 arousal 的增益（工程可动；默认 1.5=现值，零回归）。
            方向由议会定，不可改符号/单调；仅幅度可调。由 CompositeChannelDecoder 注入。
        k_coping: ⚖ 区分性 AU 对 coping 强度的增益（工程可动；默认 1.2=现值，零回归）。
            方向由议会定，不可改符号/单调；仅幅度可调。由 CompositeChannelDecoder 注入。

    热路径：纯标量（clamp/sigmoid），无 torch/LLM/网络调用。
    """
    valence, arousal = affect

    # 1) FACS AU
    if facs_extended:
        facs_au = _decode_facs_extended(
            valence, arousal, coping_potential, k_arousal=k_arousal, k_coping=k_coping
        )
    else:
        # 旧 5-AU 逐字行为（零回归）
        facs_au = _decode_facs_legacy(valence, arousal)

    # 2) 文本情绪标签：按 valence-arousal 象限映射离散词
    label = text_label(valence, arousal)

    # 3) 生理信号模拟：arousal 驱动交感输出
    # canonical_physiology=True（ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER=true）：议会 2026-07-23
    # 公式，量纲与真 PhysiologyDecoder.predict_physiology 同口径（同域同方向）：
    #   hr: 全域覆盖，Ekman/Levenson 1983 负高唤醒同样升 HR→ clamp(0.5*(1+arousal))；
    #   sc: EDA 与 |arousal| 正相关（B-2 精神·忽略效价），量纲迁 μS [0,20]；
    #   temperature_c: 高唤醒→交感血管收缩→外周降温（无 valence 项·Migliorini 2017
    #       热成像无 valence 差异；愤怒↑/恐惧↓分野属 coping_potential 第三维·VA-2D 不可区分）；
    #   pupil_mm: 删除（WESAD 无此信号·canonical 路径·议会生物席）。
    #   sc 中立偏置说明：占位 arousal=0→0μS，真 decoder 中立态~10μS，占位系统性低于真 decoder
    #   约 10μS——仅同路径内相对比较有意义，禁止跨路径绝对比较（占位/真 decoder 混用）。
    # canonical_physiology=False（默认）：逐字 legacy，零回归。
    if canonical_physiology:
        physiology: dict[str, float] = {
            "heart_rate_bpm": 50.0 + 70.0 * clamp(0.5 * (1.0 + arousal), 0.0, 1.0),
            "skin_conductance": 20.0 * clamp(abs(arousal), 0.0, 1.0),
            "temperature_c": 36.0 - 3.0 * clamp(abs(arousal), 0.0, 1.0),
        }
    else:
        physiology = {
            "heart_rate_bpm": 70.0 + 40.0 * clamp(arousal, 0.0, 1.0),
            "skin_conductance": clamp(
                abs(arousal), 0.0, 1.0
            ),  # 议会 B-2/A-P0-D：正负高唤醒均出 SCR
            "pupil_mm": 3.0 + 2.0 * clamp(arousal, 0.0, 1.0),
        }

    # 4) 语音韵律：arousal→语速/能量，valence→音高基线
    prosody = {
        "speech_rate": 1.0 + 0.5 * clamp(arousal, -1.0, 1.0),
        "pitch": 1.0
        + 0.3 * clamp(arousal, -1.0, 1.0),  # 议会 B-8/A-P0-C：F0 主随唤醒(喉肌交感张力)非效价
        "energy": clamp(0.5 + 0.5 * arousal, 0.0, 1.0),
    }

    return {
        "facs_au": facs_au,
        "text_label": label,
        "physiology": physiology,
        "prosody": prosody,
        # prosody 量纲标记（zero-link Q1 拍板 2026-07-14）：解析占位出**倍率口径**——
        # speech_rate/pitch 以 1.0 为基线、energy∈[0,1] → "ratio"。canonical 目标口径是
        # normalized [0,1]（专用 ProsodyDecoder 达标）；MCP 情感 TTS mapper 按此 tag 分支消费，
        # 收窄校验。兄弟键（非塞进 prosody 子 dict）——保 prosody 通道纯 3 值、零回归。
        "prosody_scale": "ratio",
    }


def _decode_facs_legacy(valence: float, arousal: float) -> dict[str, float]:
    """旧 5-AU 占位映射（零回归基准）。正效价→AU12/AU06；负效价→AU15/AU04；intensity∝|arousal|。"""
    # 正效价→AU12/AU6（拉嘴角/抬脸颊），负效价→AU15/AU4（压嘴角/皱眉）
    facs_au: dict[str, float] = {}
    if valence >= 0:
        facs_au["AU12"] = clamp(valence, 0.0, 1.0)
        facs_au["AU06"] = clamp(0.6 * valence, 0.0, 1.0)
    else:
        facs_au["AU15"] = clamp(-valence, 0.0, 1.0)
        facs_au["AU04"] = clamp(-0.6 * valence, 0.0, 1.0)
    facs_au["intensity"] = clamp(abs(arousal), 0.0, 1.0)
    return facs_au


def _decode_facs_extended(
    valence: float,
    arousal: float,
    coping_potential: float,
    *,
    k_arousal: float = 1.5,
    k_coping: float = 1.2,
) -> dict[str, float]:
    """13-AU 扩展映射（facs_extended=True 分支；任务 D 起含 AU17/AU26 通用 AU）。

    ⚖ 议会定方向、系数工程可动区间、待真数据校准（Gentsch 2015 fEMG 实证）。
    热路径：纯标量（clamp/sigmoid），禁 torch/LLM/网络。

    映射方向（design.md 议会裁决）：
      AU04/06/12/15/intensity：保持旧向不变（零回归基准）。
      AU05：随 arousal+ 升（高唤醒瞪眼，愤怒/恐惧共有；Ekman 1978 AU5 = 上睑抬）。
      AU07：随 arousal+ 且 v<0 升（眼部紧张，高唤醒负效价共有）。
      AU17（颏肌下巴上推）：随 −valence 主驱 + arousal 轻调制升（厌恶/悲伤；无象限守卫，
        联动 AU15 约 0.5×；议会 D·通用 AU 不进 coping 判别）。
      AU26（下颌落）：随 arousal 主驱 + 正 valence 轻压制升（恐惧/惊讶共有；无象限守卫，
        cap 因东亚惊讶少用 AU26 保守；议会 D·通用 AU 不进 coping 判别）。
      区分性 AU 仅在 (v<0, a≥0) 象限激活（象限守卫）：
        ⚠ 该唤醒必要条件是**乘性门**（a≤0 则判别 AU 恒 0），与 chat 面为治 seeking 吸引盆而设的
          arousal 压制（INTENSITY_FLOOR/AROUSAL_BASELINE/HABITUATION_TAU）叠加后，实测对抗域
          AU23 均值仅 0.009。两组设计各自结案、从未联合验证。低唤醒愤怒亚型确有实证
          （Kuppens et al. 2007，DOI:10.1080/02699930600859219）。改动本守卫或上述 arousal 旋钮
          之前须先跑议会 D3 的联合可达性网格（先判定属参数标定还是机制缺陷，勿直接调参）。
        AU23（口轮匝肌唇紧）：coping>0→高（愤怒对抗准备；Carver & Harmon-Jones 2009）。
          ⚠ 跨文化注：Cordaro 2018 跨文化核心愤怒为 AU4+AU7，AU23 未入核心（~29% 出现率）；
          此占位依 Ekman & Friesen 1978 + Scherer CPM 设计，待真权重从数据学习后修正。
        AU01/AU02（额肌扬眉）：coping<0→高（恐惧，须联动同升；Ekman 1978）。
        AU20（笑肌横拉唇）：coping<0→高（恐惧；Ekman 1978 AU20 = 唇横拉）。
      连续映射：用 relu(x)·abs(v)·|a| 风格，无 ±0.3 硬阈值；
        push 层（锥体外路泄漏）是连续渐变的（Gentsch 2015 fEMG 连续测量）。

    Args:
        k_arousal: ⚖ AU05/07 对 arousal 的增益（工程可动；默认 1.5）。
            方向由议会定，不可改符号/单调；仅幅度可调（via ZERO_FACS_K_AROUSAL env）。
        k_coping: ⚖ 区分性 AU 对 coping 强度的增益（工程可动；默认 1.2）。
            方向由议会定，不可改符号/单调；仅幅度可调（via ZERO_FACS_K_COPING env）。
    """
    # ── 系数（⚖ 议会定方向、工程可动区间；方向不可私拍，仅幅度可调）──
    K_AROUSAL: float = k_arousal  # ⚖ AU05/07 arousal 增益（经参数注入，默认 1.5）
    K_COPING: float = k_coping  # ⚖ 区分性 AU coping 增益（经参数注入，默认 1.2）

    facs_au: dict[str, float] = {}

    # ── 旧向不变通道（零回归基准）──
    if valence >= 0:
        facs_au["AU12"] = clamp(valence, 0.0, 1.0)
        facs_au["AU06"] = clamp(0.6 * valence, 0.0, 1.0)
    else:
        facs_au["AU15"] = clamp(-valence, 0.0, 1.0)
        facs_au["AU04"] = clamp(-0.6 * valence, 0.0, 1.0)
    facs_au["intensity"] = clamp(abs(arousal), 0.0, 1.0)

    # ── 新增共有通道（arousal 驱动，无象限守卫）──
    # AU05：上睑抬，随 arousal+ 升（愤怒/恐惧共有，连续）
    # ⚖ 议会定方向：高唤醒→瞪眼（Ekman 1978；Gentsch 2015 AU5 出现率随 arousal 线性升）
    arousal_pos = max(0.0, arousal)
    facs_au["AU05"] = clamp(K_AROUSAL * arousal_pos, 0.0, 1.0)

    # AU07：睑紧，随 arousal+ 且 v<0（高唤醒负效价共有，连续）
    # ⚖ 议会定方向：负效价高唤醒（愤怒/恐惧均有）→ 眼周肌紧（Gentsch 2015 AU7）
    neg_valence_gate = max(0.0, -valence)  # v<0 时 >0，v≥0 时 =0（连续软门控 = relu(-valence)）
    facs_au["AU07"] = clamp(K_AROUSAL * arousal_pos * neg_valence_gate, 0.0, 1.0)

    # AU17：颏肌下巴上推，厌恶(主)/悲伤(次)。−valence 主驱 + arousal 轻调制，联动 AU15（约 0.5×）。
    # ⚖ 议会 D 定方向（Baird 2024：AU17~arousal 显著 p=.003 / valence 负向趋势；Gentsch 2015：
    #   goal-obstructive 组、非 coping 符号判别）——**通用 AU、无象限守卫（厌恶/悲伤跨低-高唤醒）、
    #   不进 coping 判别**。系数工程可动、待真数据校准（走内置默认，不拉 env 全链）。
    K_AU17_V = 0.5  # ⚖ −valence 主驱增益（联动 AU15 的 ~0.5×，互补非冗余）
    K_AU17_A = 0.2  # ⚖ arousal 轻调制增益（k_v ≫ k_a）
    facs_au["AU17"] = clamp(K_AU17_V * neg_valence_gate + K_AU17_A * arousal_pos, 0.0, 1.0)

    # AU26：下颌落，恐惧+惊讶共有。arousal 主驱 + 正 valence 轻压制（惊喜开口＜惊恐），无象限守卫。
    # ⚖ 议会 D 定方向（Baird 2024：AU26 与 v/a 线性相关不显著→占位给理论方向、真脸细节留真模型学；
    #   Cordaro 2018：东亚惊讶少用 AU26→cap 保守）——**通用 AU、不进 coping 判别**。系数工程可动。
    K_AU26 = 1.5  # ⚖ arousal 主驱增益
    AU26_V_SUPPRESS = 0.4  # ⚖ 正 valence 压制系数（惊喜时开口幅度小于惊恐）
    AU26_CAP = 0.6  # ⚖ 跨文化保守上限（东亚惊讶少用 AU26）
    pos_valence = max(0.0, valence)
    facs_au["AU26"] = clamp(
        K_AU26 * arousal_pos * (1.0 - AU26_V_SUPPRESS * pos_valence), 0.0, AU26_CAP
    )

    # ── 区分性 AU：仅 (v<0, a≥0) 象限激活（象限守卫）──
    # 愤怒(coping>0) vs 恐惧(coping<0) 分野在 Scherer CPM + Gentsch 2015 实证
    in_quadrant = valence < 0.0 and arousal >= 0.0  # 高唤醒负效价象限
    if in_quadrant:
        # 共有强度权重：|v|·|a| 让信号在象限边界自然归零（连续，无硬阈值）
        va_weight = abs(valence) * abs(arousal)

        # AU23：愤怒对抗准备，coping>0 时高
        # ⚖ 连续映射：relu(coping)·va_weight·K_COPING；coping≤0 时输出自然为 0
        # ⚠ 跨文化注（Cordaro 2018）：AU23 非跨文化核心愤怒 AU，此处依 Ekman 1978 + CPM 占位；
        #   待真权重从 AffectNet/DISFA 数据学习后修正（Q3）。
        coping_pos = max(0.0, coping_potential)  # relu，coping<0 时=0
        facs_au["AU23"] = clamp(K_COPING * coping_pos * va_weight, 0.0, 1.0)

        # ── fear-AU 段与 WARN-3 fear 门正交（议会 2026-07-21 B-facs-fear·PASS）──
        # 表情层 fear-AU（AU01/02/20·coping<0 驱动）由 facs_extended+coping_potential_enabled
        # 双层容量门独立治理；WARN-3 fear 专属门（`AffectState.fear_domain_enabled`）治标签/符号层，
        # 与本层正交、不在此叠门——表情忠实兑现 coping_potential 连续映射，不在运动解码层再做域归类。
        # 依据：面部运动系统（CN VII 锥体外路自发通路）与情绪判定层临床双向解离（Rinn 1984；
        # Gothard 2014）；防御行为输出≠情绪意识（LeDoux & Brown 2017）；AU 非情绪类别等价、
        # fear-AU 配置一致性在全情绪类别中唯一低于偶然（Barrett et al. 2019）。
        # ⚠ 悬置 B-facs-fear-unlock：fear 域正式解锁（B-fe-unlock）时须重裁本段是否加 fear 门。
        # ⚠ 悬置 B-AU04：Ekman fear 原型=AU1+2+4+5+7+20+26，本段缺 AU04（corrugator·fear vs
        #   surprise 解剖区分件）→当前组合与 surprise 重叠，是否加 coping 驱动 AU04 待议会。
        # AU01/AU02：恐惧扬眉，coping<0 时高（联动同升，两 AU 值相等）
        # ⚖ 连续映射：relu(-coping)·va_weight·K_COPING；coping≥0 时输出自然为 0
        coping_neg = max(0.0, -coping_potential)  # relu(-coping)
        fear_score = clamp(K_COPING * coping_neg * va_weight, 0.0, 1.0)
        facs_au["AU01"] = fear_score
        # 联动同升（Ekman 1978）；⚠ Gentsch 2015（gambling task）仅报 coping×effort 交互效应·
        # frontalis 主效应不显著·且未测 AU20→此处方向属 CPM 理论预测，非 Gentsch 主效应实证（A1）。
        facs_au["AU02"] = fear_score

        # AU20：笑肌横拉唇，恐惧（coping<0）时高
        # ⚖ 连续映射：同 AU01/02 的 coping 负部驱动（Ekman 1978 AU20 = 恐惧特有）
        facs_au["AU20"] = clamp(K_COPING * coping_neg * va_weight, 0.0, 1.0)
    else:
        # 象限外：区分性 AU 归零（防意外激活）
        facs_au["AU23"] = 0.0
        facs_au["AU01"] = 0.0
        facs_au["AU02"] = 0.0
        facs_au["AU20"] = 0.0

    return facs_au


def fast_survival_prior(
    features: list[float],
    *,
    commensurable: bool = False,
    arousal_floor_fix: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """快生存流（快速皮层下/防御回路，仿 LeDoux 生存回路思路）：从原始特征出粗 (μ, Π)。

    亚符号、低精度、最快：只用目标一致性符号定效价方向、强度定唤醒，不做 OCC 多维细评
    （那是慢评价流的事）。features = [goal, standard, attitude, intensity]（PerceptionAgent）。
    精度固定为 SURVIVAL_PRECISION（粗快=不确定），逐维返回。纯函数、无副作用。
    注（议会 B-1/M2）：具体解剖通路（上丘→丘脑枕→杏仁核"捷径"）在人类证据仍有争议
    （Pessoa & Adolphs 2010, NRN 11:773；LeDoux & Brown 2017），此处仅作快速显著性评估的
    工程近似，不承诺特定解剖底物。

    `commensurable=True`（第四轮议会量纲齐次化，默认关）：把常数 0.4 换成与 `occ_prior`
    同构的 `conf → σ → 1/σ²` 链路，但截距/斜率/上限都远低于慢评价流
    （0.1/0.1/0.5 vs 0.3/0.5/1.0），使「粗快=不确定」这一设计意图**在同一把尺子上**
    成立——Π_survival∈[4.94, 6.25]，恒弱于 Π_appraisal∈[8.16, 200]。

    `arousal_floor_fix=True`（议会 D5「失真必改」，默认关=逐字旧行为）：去掉 arousal 的
    **0.5 地板**。`I=0` 时仍给 0.5 = 把「没有证据」编码成「确定的中等唤醒」——对无信号撒谎。
    亚皮层通路响应幅度随刺激分级（McFadyen 2019），无威胁时应趋于基线。
    与 valence 项 `clamp(0.6*goal, -1, 1)`（本就无常数底数）内部一致。
    ⚠ 须与 `gate_fusion` **共用同一 default-off 门控**，杜绝「只修地板不改架构」这种未评审的中间态。
    ⚠ 这是**装配阶段**的 μ 污染：`(streams → post)` 形状的锚点按构造抓不到它（会从已污染的
    流列表重新推导参照），故另有流级锚点 C 专门承接。
    """
    goal = features[0] if features else 0.0
    intensity = features[3] if len(features) > 3 else 1.0
    valence = clamp(0.6 * goal, -1.0, 1.0)  # 粗：只取目标符号
    if arousal_floor_fix:
        arousal = clamp(0.5 * abs(intensity), 0.0, 1.0)  # 无信号 → 趋基线，不撒谎
    else:
        arousal = clamp(0.5 + 0.5 * abs(intensity), 0.0, 1.0)  # 威胁/显著 → 高唤醒（旧·带地板）
    if commensurable:
        conf = clamp(
            SURVIVAL_CONF_BASE + SURVIVAL_CONF_GAIN * abs(intensity), 0.0, SURVIVAL_CONF_CAP
        )
        prec = 1.0 / max(MIN_SIGMA, 0.5 * (1.0 - conf)) ** 2
        return (valence, arousal), (prec, prec)
    return (valence, arousal), (SURVIVAL_PRECISION, SURVIVAL_PRECISION)


def stream_salience(mu: tuple[float, float], precision: tuple[float, float]) -> float:
    """门控分数：偏离中性的幅度 × 平均精度，`|μ|·Π̄`。偏离越远、精度越高 → 越该进入全局广播。纯函数。

    ── 引文归属（议会 2026-07-30 订正：此前锚点错位）──
    两个乘子出自**两套不同**的理论传统，此前整条公式被挂在显著网络文献名下，是错的：
      · `hypot(μ)`（离中性距离 = 强度）—— core affect 的强度定义
        （Russell 2003, https://doi.org/10.1037/0033-295X.110.1.145）。
      · `Π̄`（精度作为门控权重）—— 主动推理的「注意即精度推断」
        （Feldman & Friston 2010, https://doi.org/10.3389/fnhum.2010.00215；
        内感受侧见 Barrett & Simmons 2015 EPIC, https://doi.org/10.1038/nrn3950）。
      · 「显著网络（前岛叶+dACC）」（Menon & Uddin 2010,
        https://doi.org/10.1007/s00429-010-0262-0）是这套计算的**神经解剖底物**，
        该文给的是「做行为/内稳态相关性检测」的功能定位，**并未给出**本公式；
        故不得再把 `|μ|·Π̄` 记作「= 显著网络门控分数（Menon & Uddin）」。
      · 亦**不**对应 Scherer CPM 的 relevance/novelty check——那问的是「与目标是否相关」，
        不是「测得准不准」。

    ── ⚠ 两条已知简化（议会 2026-07-30 记录，本轮不改数值行为）──
    1. **非 canonical 形式**：自由能原则下的显著性是**逐维**精度加权预测误差（`Σ_d Π_d·ε_d²`），
       而本式把「多大」（`hypot`，两维等权、不看各自精度）与「多确信」（`mean(Π)`，算术平均）
       解耦成两个标量相乘 ⇒ 丢了「某一维精度很高时该维应主导显著性」这一性质。
    2. 🛑 **`mean_precision` 对结构性一维流有约 2.83× 的隐藏折扣**：
       对两维精度都为 Πa 的「公平」二维流，最坏 salience = `√2·Πa`；
       而 physio 流经 M2 后 Πv 恒为 MIN_PRECISION ⇒ 最坏 salience = `0.5·Πa`，二者相差 `2√2 ≈ 2.83`
       （`√2` 来自维度缺失、`2` 来自把近零分量也计入平均）。
       该折扣是**公式副作用、不是有意设计的跨流公平性保证**，也不是 `SALIENCE_THRESHOLD` 的反推来源
       （它与 `0.5/0.18 ≈ 2.78` 的接近是巧合）。
       ⚠ **若日后把 `mean_precision` 改成 `max` 或按有效维度数归一化，该折扣会静默消失且无测试报警**
       —— 改本函数前请重新核算这个折扣是否还成立。
    """
    deviation = math.hypot(mu[0], mu[1])
    mean_precision = 0.5 * (precision[0] + precision[1])
    return deviation * mean_precision


def _score_streams(
    streams: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> list[tuple[str, tuple[float, float], tuple[float, float], float]]:
    """给每条流打 salience 分，返回 (name, μ, Π, salience)。"""
    return [(name, mu, prec, stream_salience(mu, prec)) for name, mu, prec in streams]


def _select_fired(
    scored: list[tuple[str, tuple[float, float], tuple[float, float], float]],
    *,
    threshold: float,
    survival_fallback: bool,
    soft_beta: float | None,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """今天 `ignite()` 的筛选逻辑，原样抽出供数值通路与报告通路**共用**。

    抽成 helper 而非各写一遍，是为了保证 `gate_fusion=True` 时 `ignite()` 与
    `report_ignited()` 的输出**逐值相等**（零回归的必要条件，design D13）——
    两处独立实现同一套兜底分支迟早会分叉。
    """
    if soft_beta is not None:
        # 软门控分支：所有流参与，精度按 logistic gate 调制
        out: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for name, mu, prec, sal in scored:
            gate = sigmoid(soft_beta * (sal - threshold))
            out.append((name, mu, (prec[0] * gate, prec[1] * gate)))
        return out

    # 硬 step 分支（soft_beta=None）
    fired = [(name, mu, prec) for name, mu, prec, s in scored if s >= threshold]
    if fired:
        return fired
    if survival_fallback:
        surv = next(((n, m, p) for n, m, p, _ in scored if n == "survival"), None)
        if surv is not None:
            return [surv]
    top = max(scored, key=lambda item: item[3])
    return [(top[0], top[1], top[2])]


def ignite(
    streams: list[tuple[str, tuple[float, float], tuple[float, float]]],
    *,
    threshold: float = SALIENCE_THRESHOLD,
    survival_fallback: bool = False,
    soft_beta: float | None = IGNITION_BETA,
    gate_fusion: bool = True,
    exclude_physio_fusion: bool = True,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str]]:
    """GNW ignition：salience ≥ threshold 的流点燃进入全局广播，亚阈流停留局部。

    若无流过阈（全弱刺激）：
    - `survival_fallback=False`（默认）：保留 salience 最高者（逐字旧行为，零回归）。
    - `survival_fallback=True`：优先保留 name=="survival" 的流作兜底（皮层下常播，
      与 GNW 一致——亚阈刺激停留局部、但皮层下生存信号始终广播）；若 streams 中无
      survival 流，则退回 max by salience（不崩）。

    有流过阈时两分支行为完全一致，`survival_fallback` 不影响结果。

    软门控（议会 2026-07-02 Item 2 · [工程可动] · 已实现）：
    `soft_beta=None`（默认）→ 逐字硬 step 旧行为，零回归。
    `soft_beta` 非 None → 对每条流用 logistic gate 软化 all-or-none：
        gate(sᵢ; θ, β) = σ(β·(sᵢ − θ))
        π_effᵢ = (prec[0]·gate, prec[1]·gate)
    所有流均参与 fuse_terms（gate>0 自动保证不空播）；高 salience 流精度大量保留、
    亚阈流精度压至近 0（连续近似 Dehaene & Changeux 2011 bifurcation 双稳）。
    β<1 无意义（gate 趋 0.5 成均匀融合），推荐区间 [20,50]（神经/数学席交集）。
    复用本模块既有 `sigmoid`。

    ── `gate_fusion`（议会第三轮 D1，默认 True=门关，逐字旧行为）──
    `True`：硬门同时决定「谁进 fuse_terms」与「谁可报告」——即今天的行为。
    `False`：**硬门从数值通路上摘下来，只留报告通路**。`fusion_terms` 收全部流的原生 (μ, Π)，
    不乘任何 gate/D 因子；「哪些流点燃」改由 `report_ignited()` 单独回答。
    神经席裁定：GNW ignition = 「什么内容变得**可报告**」，不是「谁计算数值」；
    **阈下不点燃 ≠ 阈下零影响**。

    ── 🛑 `exclude_physio_fusion` 的作用域**只有 `gate_fusion=False` 分支**（D7 的已知缺口）──
    本参数的过滤**只在门开分支执行**；门关（默认）分支调完 `_select_fired` 即提前 return，
    根本走不到它。故 D7「physio 不进数值通路」这个跨仓承诺在门关路径下**并未由本开关兑现**：
    - 门关 + 硬门（`soft_beta=None`，默认）：physio 进不来靠的是**它自己的低 Πa**
      （Zero_MCP 线上载荷 salience 上界 0.088 < 阈值 0.18），不是 D7。把 Πa 抬到
      ≥0.359 即可在此路径下自行过阈。该边界**对方侧现已落成运行期守卫**（其 M8 在出网
      收口点 `build_external_priors_override` 现算最坏 salience，达阈即 raise），已不再
      只是自律；但那是**对方仓内的单边守卫**，我方侧仍无结构约束——不经该收口点的入口
      （我方 chat 面、测试夹具、其它 MCP client）照样能把这样的载荷送过阈。
      回归锁：`tests/test_external_priors.py::TestPhysioSelfIgniteExposureOnDefaultPath`
      （该界现算、双向断言曝露面仍在）。
    - 门关 + 软门（`soft_beta` 非 None）：**全部流含 physio 一律进 `fuse_terms`**（精度乘
      logistic gate），既无阈值筛除也无 D7 —— 这是**真实旁路**。默认 `IGNITION_BETA=None`
      故生产未触发。⚠ 旁路面**比早前记的小**：`ignition_beta` 现已在
      `_MCP_GOVERNANCE_GATED_FLAGS` 内，client 经 `open_session(config={"ignition_beta": ...})`
      传它会被**静默忽略**；今天只有部署端 env `ZERO_MCP_IGNITION_BETA` 能打开软门。

    **收口条件**（议会 design A4 要求记录时间窗，此前遗漏）：把前缀过滤提到 `if gate_fusion:`
    之前、对三条路径同施。属**行为变更须走议会门**，故本轮只做记录 + 特征化测试。
    ⚠ 该改动**不会**破 `tests/test_ignition_soft_gate.py`（该文件不含任何 physio 前缀流名）；
    实测会红的是 `tests/test_ignition_gate_fusion.py` 的 13 条：`::test_scenario_matrix` 的
    12 条软门格（4 present × 3 n_external × soft_beta=20.0）+ 门关硬门路径上的
    `::test_d7_gap_hard_gate_high_salience_physio`。两者都是**双向**断言，收口那天会红在
    「缺口已被堵上」而非「缺口仍在」，照消息把 `_D7_GAP_OPEN` 翻成 False 即可。
    ⚠ 堵缺口当天仍须 ping Zero_MCP，但**理由已变**（2026-07-30 据对方来件订正）：
    此前这里写「其 `test_soft_gate_bypasses_physio_exclusion` 锚在 `_select_fired` 函数体上
    ⇒ 我方只改 `ignite()` 时它不会变红而是静默失去刻画能力」——**该描述已过期**：
    对方 2026-07-29 已按我方建议把**主锚点移到 `ignite()` 的门判之前**（即本收口方案的落点），
    `_select_fired` 那两条降级为**副锚点**保留（覆盖「把过滤下沉进 helper」这条另一种修法），
    并配了两条正控防断言退化成恒真。⇒ **收口时不必再为对方的锚点位置做额外动作。**

    🛑 但仍有**一格盲点**（对方如实交底、我方照记，不声称已覆盖）：若把过滤抽成
    **不透明 helper**（形如 `streams = _drop_physio(streams)`，开关从模块级读、**不作实参传**）
    并在门判前调用，对方守卫**看不见**（其 11 态判别力矩阵该格**实测为绿**）。
    ⇒ 收口若采用不透明 helper 形态，**ping 是唯一的跨仓信号**；采用显式传参形态则对方会红。
    写了盲点不等于覆盖了盲点——这句照抄对方原话，因为它同样适用于我方自己的守卫。

    ⚠ **本函数的两个返回值恒对齐**（同一次筛选产出、一一对应）——调用方可安全
    `zip(..., strict=True)`。这是 BLOCK 1 的实质保证；不需要第三个返回值，见 design D13。

    streams = [(name, μ, Π), ...]；返回 (供 fuse_terms 的 [(μ, Π)], **与之对齐的**流名列表)。
    纯函数、无副作用。
    """
    scored = _score_streams(streams)
    if gate_fusion:
        # 门关（默认）：逐字旧行为——硬门/软门的产物同时充当融合项与报告项。
        selected = _select_fired(
            scored, threshold=threshold, survival_fallback=survival_fallback, soft_beta=soft_beta
        )
        return [(mu, prec) for _, mu, prec in selected], [name for name, _, _ in selected]

    # 门开：全流原生 (μ, Π) 进融合，不乘任何 gate/D 因子。
    fusion = [(name, mu, prec) for name, mu, prec, _ in scored]
    if exclude_physio_fusion:
        # D7 跨仓承诺（默认排除）。⚠ **依据按前缀分三档，不是一条证据覆盖五个前缀**
        # （Zero_MCP 2026-07-29 明确要求记账时别把无证据的前缀计为「有证据支持的排除」）：
        #   · eda_*  —— WESAD 真被试实测其 arousal 与唤醒**系统性反号**（v1，正确率 1/5）。
        #              其 v2 已改口径（判别力 10/10、无反号），但 v2 **仅在其未合并分支上**，
        #              其 main 仍跑已证伪的 v1 → 解除以「v2 合入其 main」为触发点。
        #   · hrv_*  —— **从无支持排除它的独立证据**，是被瞄准 EDA 反号的一刀切前缀规则顺带
        #              扫进来的。其审计：判别 4/5、无符号反转，问题是**读数吵**（漂移中位数
        #              0.72）不是方向错 → 正确表达是低精度而非排除，解除排期早于 eda_*。
        #   · pupil_* / scr_* —— **零载荷、零实测依据**：scr_amplitude 随其 EDA v1 一并删除，
        #              瞳孔通道从未实现。留着不碍事，但**不构成有证据支持的排除**。
        # 由我方单边可控。⚠ 本过滤只在门开分支执行，门关路径下 D7 不生效——见 ignite docstring。
        fusion = [t for t in fusion if not t[0].lower().startswith(_PHYSIO_PREFIXES)]
    if streams and not fusion:
        # D12：违反前置条件（传入的流**全是**被排除流）。生产路径不可达——`affect_core` 恒装配
        # survival/appraisal/value 三条核心流。此处给指名道姓的报错，替掉调用方 fuse_terms
        # 里那个无信息的 ZeroDivisionError。**不做「回退全流」**——那会让被排除的 physio
        # 无声重新参与融合，直接违背对 Zero_MCP 的承诺。
        raise ValueError(
            "ignite(gate_fusion=False) 的 fusion_terms 为空：传入的 "
            f"{len(streams)} 条流全部命中 physio 排除前缀 {_PHYSIO_PREFIXES}。"
            "至少需要一条非 physio 流；若确要让 physio 参与融合，显式传 exclude_physio_fusion=False"
        )
    return [(mu, prec) for _, mu, prec in fusion], [name for name, _, _ in fusion]


def report_ignited(
    streams: list[tuple[str, tuple[float, float], tuple[float, float]]],
    *,
    threshold: float = SALIENCE_THRESHOLD,
    survival_fallback: bool = False,
    soft_beta: float | None = IGNITION_BETA,
) -> list[str]:
    """**报告通路**：哪些流「变得可报告/可内省」（GNW ignition 的正确语义归属，神经席 D4）。

    与 `ignite()` 的数值通路彻底分离——本函数的输出**不影响任何数值**，只作可解释性标签
    （`AffectState.ignited_streams`）。判据与今天逐字相同（含软门全流、硬门兜底两条分支），
    故 `gate_fusion=True` 时它与 `ignite()` 返回的流名列表**逐值相等** → 零回归。

    ⚠ 该列表**不是纯标注**：经 `supervisor.py` 落进 **USER 作用域 episode**，
    再由 `memory_recall` → `chat_driver` 注入 LLM system 上下文。改它的语义要当外部可见行为对待。
    """
    selected = _select_fired(
        _score_streams(streams),
        threshold=threshold,
        survival_fallback=survival_fallback,
        soft_beta=soft_beta,
    )
    return [name for name, _, _ in selected]


def attitude_step(
    attitude: tuple[float, float],
    stimulus: tuple[float, float],
    *,
    rate: float = ATTITUDE_RATE,
    reversion: float = ATTITUDE_REVERSION,
    setpoint: tuple[float, float] = ATTITUDE_SETPOINT,
    reversion_a: float | None = None,
    arousal_weight: float = 0.0,
) -> tuple[float, float]:
    """慢变态度/印象：按 stimulus 缓慢累积 + 向个体基线弱回归（evaluative conditioning）。

    `a' = (1-rate_eff)·a + rate_eff·stimulus − reversion·(a − setpoint)`。
    rate 小（多轮才成形）→ 长期、
    稳定、对象指向的评价（Scherer/Frijda 的 sentiment/attitude 层）。`reversion` 项是议会必改：
    无它则持续同向 stimulus 把 attitude 单调推到极端（affective homeostasis 缺失 / 慢性应激无负
    反馈，Russell 2003 · Kuppens 2010）；有它则恒定刺激下稳态 a*≈rate·s/(rate+reversion)、被钳在
    |s| 内不无限漂移，刺激停歇时缓慢回基线。`reversion=0` 退化为旧纯 EWMA（零回归开关）。

    P1-b（议会 Q2b·失真必改）：`reversion_a` 为 arousal 维**独立**回归率（默认 None → 用 `reversion`
    两维同构=逐字旧行为、零回归）。心理席主裁：attitude 是 valence 维评价，"对某人的长期唤醒基线"
    无文献先例。设 `reversion_a`≫`reversion`（纪要推荐 0.3–0.5）使 arousal 稳态 a*≈setpoint_a≈0，
    功能上令 attitude 不再累积 arousal 直流偏置，同时不改二元组结构（零 breaking）。纯函数。

    A-P2-E `arousal_weight`（默认 0.0，零回归）：高唤醒 stimulus 使累积率放大。
    `rate_eff = rate * (1 + arousal_weight * |stimulus[1]|)`（McGaugh 2004：唤醒调制记忆巩固/
    态度形成；唤醒越高越快收敛到 stimulus）。默认 0.0 → rate_eff = rate，逐字旧行为。
    """
    rev_a = reversion if reversion_a is None else reversion_a
    # A-P2-E：唤醒加权有效累积率（默认 0.0 → rate_eff = rate，零回归）
    rate_eff = rate * (1.0 + arousal_weight * abs(stimulus[1]))
    return (
        clamp(
            (1.0 - rate_eff) * attitude[0]
            + rate_eff * stimulus[0]
            - reversion * (attitude[0] - setpoint[0]),
            -1.0,
            1.0,
        ),
        clamp(
            (1.0 - rate_eff) * attitude[1]
            + rate_eff * stimulus[1]
            - rev_a * (attitude[1] - setpoint[1]),
            -1.0,
            1.0,
        ),
    )


def habituation_factor(
    exposure: int,
    tau: float,
    *,
    intensity: float = 0.0,
    sensitization_gain: float = 0.0,
    sensitization_threshold: float = 0.5,
) -> float:
    """净反应系数 η(n)：习惯化衰减 + 敏化增益双过程（Groves & Thompson 1970）。

    P2（议会 Q6·失真必改）：对同一对话对象累计曝光 `exposure` 轮，给 arousal 输入乘 η 调制。
    `tau<=0` → 返回 1.0（不衰减，零回归开关）；`exposure` 负值按 0 处理。

    A-P3-D 双过程扩展（默认参数 = 零回归）：
      η = exp(−n/τ) + sensitization_gain · max(intensity − sensitization_threshold, 0)
    强刺激（intensity 高于 sensitization_threshold）时敏化项增益，η 可 >1 表示敏化主导。
    默认 sensitization_gain=0.0 → η = exp(−n/τ)，逐字旧行为（Groves & Thompson 1970：
    弱/中性刺激习惯化主导；强/厌恶刺激敏化主导，净效应=两过程竞争结果）。
    纯函数、无 I/O、无 env——参数由上层（chat_driver）按 env 注入。
    """
    if tau <= 0.0:
        return 1.0
    hab = math.exp(-max(0, exposure) / tau)
    # A-P3-D：敏化项仅在 intensity 超过阈值时激活（默认 gain=0.0 → 旧行为，零回归）
    sen = sensitization_gain * max(intensity - sensitization_threshold, 0.0)
    return hab + sen


def emotion_decay_step(
    emotion: tuple[float, float],
    baseline: tuple[float, float],
    stimulus: tuple[float, float],
    *,
    recovery: float = EMOTION_RECOVERY,
    reactivity: float = EMOTION_REACTIVITY,
) -> tuple[float, float]:
    """快变情绪：向 baseline(=当前态度) 衰退恢复 + 被当前 stimulus 冲击（affective chronometry）。

    `e' = baseline + recovery·(e - baseline) + reactivity·stimulus`。情绪是**短时**的：
    刺激停了（stimulus≈0）则 deviation 每轮 ×recovery 衰减、几轮内回到 baseline（情绪不长期累积，
    过慢衰退=emotional inertia 病理）；持续 stimulus 则稳态 = baseline + reactivity·s/(1-recovery)。
    baseline 由慢变 attitude 给出 → 怒火退去后回到「对此人的态度」而非绝对中性。逐维钳制，纯函数。

    M13 嵌套 AR(1) 注：本函数 AR(1)（recovery≈0.4，对应 Kuppens 2010 情绪惯性度量）与慢变
    attitude 基线（attitude_step 中 AR≈0.92，1-ATTITUDE_RATE=0.92）嵌套——emotion 的有效自相关
    略高于 0.4（基线本身带历史依赖，情绪衰退目标随 attitude 漂移）。此属**已知设计意图**：
    reversion 项（attitude_step 中的弱均值回归）压制单调漂移；两层 AR 的谱半径均 <1（recovery=0.4，
    inertia≈0.92），系统收敛、非病态吸引子（Kuppens et al. 2010 Psychol. Sci. 21(7):984-991，
    PMC2901421）。若需调整情绪记忆时间尺度，优先改 recovery 参数（persona.recovery），
    而非改 AR 结构本身。
    """
    return (
        clamp(
            baseline[0] + recovery * (emotion[0] - baseline[0]) + reactivity * stimulus[0],
            -1.0,
            1.0,
        ),
        clamp(
            baseline[1] + recovery * (emotion[1] - baseline[1]) + reactivity * stimulus[1],
            -1.0,
            1.0,
        ),
    )


def precision_reconcile(
    e_star: tuple[float, float],
    e_precision: float,
    lang_affect: tuple[float, float],
    *,
    lang_precision: float = LANG_BASE_PRECISION,
) -> tuple[float, float]:
    """精度加权再入（替代固定 RECONCILE_WEIGHT 的中点拉拢）：内核 e* 与语言反推情感
    按各自精度融合——高精度内核抗语言拉拢、低精度内核让步。逐维钳制 [-1, 1]，纯函数。

    e_precision=语言侧精度时退化为中点（≡ reconcile_affect weight=0.5）。
    """
    den = max(MIN_PRECISION, e_precision + lang_precision)
    v = clamp((e_precision * e_star[0] + lang_precision * lang_affect[0]) / den, -1.0, 1.0)
    a = clamp((e_precision * e_star[1] + lang_precision * lang_affect[1]) / den, -1.0, 1.0)
    return (v, a)


def suppress_expression(
    affect: tuple[float, float],
    *,
    factor: float,
) -> tuple[float, float]:
    """表达抑制（Gross 1998 response-focused regulation）：按 factor 压制输出表达幅度。

    A-P2-C：`factor∈[0,1]` 缩放 (valence, arousal) 的表达幅度，语义＝压制末端输出，
    不改变内部体验 e*（与 `reappraise` 区分：重评改构念/体验，抑制仅压表达通道）。
    调用方作用于表达通道而非内核 e*，内核保持不变（Gross 1998 process model）。
    factor=1.0 → 无抑制（幅度不变）；factor=0.0 → 完全抑制（输出归零）。
    逐维乘以 factor 后钳制 [-1, 1]，纯函数、无 I/O、无副作用。
    """
    v = clamp(affect[0] * factor, -1.0, 1.0)
    a = clamp(affect[1] * factor, -1.0, 1.0)
    return (v, a)


def cortisol_step(
    cortisol: float,
    delta_t: float,
    *,
    tau_decay: float,
    impulse: float = 0.0,
    cap: float = 1.0,
) -> float:
    """HPA 皮质醇慢回路一步更新（精确 ZOH 离散化，P3 1-B design.md §一·数学席）。

    方程（Zero-Order Hold 精确离散化）：
        α = exp(−max(0, delta_t) / tau_decay)
        c_new = clamp(α · cortisol + impulse, 0.0, cap)

    **禁 Euler** `c·(1−Δt/τ)`：Δt > τ 时乘子变负、Δt > 2τ 时发散（用户隔 2h 回复
    即 Δt≈7200s > τ≈5400s 便崩）。精确 ZOH α=exp(−Δt/τ)∈(0,1) 任意正 Δt 无条件有界。

    参数：
        cortisol: 当前皮质醇水平 ∈ [0, cap]。
        delta_t:  自上次更新经过的秒数（由编排层节点计算后传入，**此函数体内无时钟调用**）。
                  delta_t < 0 防御：按 0 处理（等价未经过时间，皮质醇不衰减）。
        tau_decay: 衰减时间常数（秒）；生物席推荐 ≈5400s（血浆半衰期 ~70min / ln2）。
        impulse:   本步脉冲注入量（触发应激时由 cortisol_trigger 给出；否则 0.0）。
        cap:       皮质醇上界（归一 1.0；对应肾上腺分泌生理上限，防 runaway 兜底）。

    红线（code-reviewer 复核）：
        - 体内无任何 datetime.now() / time.time() 调用（破可复现·见 design §五 CS 红线）。
        - 纯标量 Python，无 torch / LLM（守热路径红线）。
        - cap 确保有界（design §一有界性证明）。
    """
    alpha = math.exp(-max(0.0, delta_t) / tau_decay)
    return clamp(alpha * cortisol + impulse, 0.0, cap)


def cortisol_trigger(
    goal_congruence: float,
    intensity: float,
    *,
    theta_goal: float = CORTISOL_THETA_GOAL,
    theta_intensity: float = CORTISOL_THETA_INTENSITY,
    impulse: float = CORTISOL_IMPULSE,
) -> float:
    """HPA 触发判据：目标受阻 + 高强度 → 皮质醇脉冲（触发解耦，P3 1-B design.md §三）。

    判据（Dickerson & Kemeny 2004，208 项元分析；d=0.93）：
        不可控性（goal_congruence < −theta_goal）AND 高强度（intensity > theta_intensity）
        → HPA 激活，返回 impulse；否则 0.0。

    **只读 appraisal 输入**（goal_congruence / intensity），**绝不读 arousal / emotion 状态**。
    这是触发解耦（机制 A）的核心：∂I/∂c ≡ 0 → 回路数学开环 → 线性稳定收敛，防正反馈
    runaway（cortisol↑→arousal↑→触发→cortisol↑ 闭合正反馈在 v1 被此解耦切断）。

    生理依据（生物席）：真实 HPA 触发是杏仁核/PVN 对外部威胁的评价，非当前皮质醇水平；
    纯高唤醒可控任务 HPA 反应弱（Dickerson & Kemeny 2004 分组对比 d 值差异）。

    参数：
        goal_congruence: appraisal 输入（Stimulus.goal_congruence），非 arousal 状态。
        intensity:       appraisal 输入（Stimulus.intensity），非 emotion 状态。
        theta_goal:      目标不一致阈值（推荐 0.3；实际走 env ZERO_CORTISOL_THETA_GOAL）。
        theta_intensity: 强度阈值（推荐 0.5；实际走 env ZERO_CORTISOL_THETA_INTENSITY）。
        impulse:         触发时注入量（推荐 0.6-0.8；实际走 env ZERO_CORTISOL_IMPULSE）。

    引文：Dickerson, S. S., & Kemeny, M. E. (2004). Acute stressors and cortisol responses.
    *Psychol. Bull.* 130(3):355-391. https://doi.org/10.1037/0033-2909.130.3.355
    """
    if goal_congruence < -theta_goal and intensity > theta_intensity:
        return impulse
    return 0.0


# ── 外部多模态先验流展开（议会 2026-07-15 M1–M6；PRP 外部多模态先验流注入口）──
# ExternalPrior 类型注解 import 见文件顶部 TYPE_CHECKING 块（同层·叶子协议·无反向依赖）。
# 生理流前缀集合（M2·议会生物席强制）：以任一前缀开头的流名视为生理信号流，
# 无条件覆写 Πv=MIN_PRECISION —— 依据是「**单一通道原始读数**区分效价的能力有限」，
# 见 expand_external_priors docstring 的 M2 段（议会 2026-07-30 已订正该处的机制表述与引用方向）。
_PHYSIO_PREFIXES: tuple[str, ...] = ("physio", "eda", "hrv", "pupil", "scr")


def expand_external_priors(
    external_priors: list[ExternalPrior],
    *,
    precision_cap: float,
    max_streams: int,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """外部多模态先验流展开 + 防御性校验，返回可直接 extend 进 streams 的列表。

    M6（数学席·流数上界）：len(external_priors) > max_streams → raise ExternalPriorError。

    ⚠ 本函数的所有校验失败一律抛 **`ExternalPriorError`**（`ValueError` 子类，向后兼容），
    **不是**裸 `ValueError`。理由是归责可辨：边界层 `server.py` 据此把「真的是 client 传参
    不对」与「内核自己出错了」分开报——后者若被贴成前者，client 照着改传参永远改不好
    （议会 2026-07-29 第五轮校验 §四-5）。新增校验请沿用该类型。
    校验精度来自 MCP payload，Zero 只校验+防御——不硬编码各模态精度。

    形状良构校验（澄清 2）：每条须 (str, (float, float), (float, float))；
    name 必须是 str，mu/prec 各须是 2 元 float tuple；防 MCP 传标量精度或错元数。

    M3（CS 席·fail-fast 放展开处）：
      - Πv > 0 且 Πa > 0，否则 raise ValueError（精度须正）。
      - Πv ≤ precision_cap 且 Πa ≤ precision_cap，否则 raise ValueError。
      fail-fast 在此处（非 SessionConfig）：external_priors 是每轮 state_overrides
      动态内容，SessionConfig 只能管会话级默认，管不到每轮 payload（design.md M3）。

    M2（生物席强制·唯一「失真必改」）：name 以 _PHYSIO_PREFIXES 任一前缀开头
    （小写比较）→ 无条件覆写 Πv = MIN_PRECISION，即便 MCP 给了有意义值也归零。
    依据（议会 2026-07-30 订正两处，原文有两个错，结论不变）：
      · ❌ 原写「EDA/HRV/瞳孔**只编码交感**唤醒输出」——**机制说错了**：
        RMSSD 是**迷走/副交感**张力指标（Shaffer & Ginsberg 2017,
        https://doi.org/10.3389/fpubh.2017.00258；Appelhans & Luecken 2006,
        https://doi.org/10.1037/1089-2680.10.3.229），瞳孔受**交感+副交感双重**调制，
        只有 EDA（外泌汗腺）是交感单支。三者的共同点是**弱效价特异性**，不是「都是交感」。
      · ❌ 原文把 Kreibig 2010 整篇当作「生理对效价盲」的靠山——**引用方向反了**：
        该综述的核心论点恰恰是**反对**「未分化唤醒」模型、
        主张 *considerable ANS response specificity*。
      · ✅ 站得住的窄 claim 是：「**单一通道的原始读数**区分效价的能力有限」
        （EDA 侧最对题的是 Bradley, Codispoti, Cuthbert & Lang 2001,
        https://doi.org/10.1037/1528-3542.1.3.276 —— 电导随**唤醒**变、效价由面部通道携带；
        另见 Mauss & Robinson 2009, https://doi.org/10.1080/02699930802204677）。
      · ⚠ 诚实附注：HRV **并非严格对效价盲**——Kreibig 2010 自己报告负性情绪普遍伴 HRV 降低、
        而正性情绪的影响含混，即存在一小块与威胁评价挂钩的不对称信息。
        判定为「简化」而非「失真」：该残留信息在 HRV 实测 σ≈1.4–1.8 的信噪比下量不出来。
    给 valence 精度 = 主动注入偏差（design.md §二·生物席强制·收敛 a）。

    返回与 `affect_core` 里 streams 装配处（`if state.workspace_enabled:` 分支内）
    类型完全一致的列表，可直接 extend（M1）。
    不进 occ_prior/survival 入口（design.md 受约束方案 c）。

    引文：Kreibig S.D. (2010). Biol. Psychol. 84(3):394-421.
    https://doi.org/10.1016/j.biopsycho.2010.03.010
    ⚠ 保留该条只为可追溯（它是 M2 最初的立项引文），但**它支持的不是「生理对效价盲」**——
    引用方向的订正与真正对题的文献见上方 M2 的「依据」段（议会 2026-07-30）。
    设计决策见 design.md M1–M6（议会 2026-07-15）；本轮订正见
    notes/2026-07-30-design-options-precision-criterion-vs-ignition-threshold.md 的「议会裁定」段。
    """
    # M6：流数上界
    if len(external_priors) > max_streams:
        raise ExternalPriorError(
            f"external_priors 条数 {len(external_priors)} 超过 max_external_streams={max_streams}；"
            f"请检查 MCP 传参（ZERO_MAX_EXTERNAL_STREAMS 调大或减少注入流数）"
        )

    result: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for i, prior in enumerate(external_priors):
        # 形状良构校验（澄清 2）
        if (
            not isinstance(prior, tuple)
            or len(prior) != 3
            or not isinstance(prior[0], str)
            or not (isinstance(prior[1], tuple) and len(prior[1]) == 2)
            or not (isinstance(prior[2], tuple) and len(prior[2]) == 2)
            or not all(isinstance(x, (int, float)) for x in prior[1])
            or not all(isinstance(x, (int, float)) for x in prior[2])
        ):
            raise ExternalPriorError(
                f"external_priors[{i}] 形状不合法：须为 (str, (float,float), (float,float))，"
                f"实际为 {prior!r}；请检查 MCP as_zero_streams() 输出格式"
            )
        name: str = prior[0]
        mu: tuple[float, float] = (float(prior[1][0]), float(prior[1][1]))
        pi_v: float = float(prior[2][0])
        pi_a: float = float(prior[2][1])

        # M7：μ 域校验（议会 2026-07-28）。契约（external_prior.py）声明 μv/μa ∈[-1,1]，
        # 但此前只是注释、运行期不校验——安全边界实际靠 MCP 侧 ModalityPrior 自律维持。
        # 按 Parse-don't-validate 在解析处一次性校验到位：越界 μ 会直接抬高 stream_salience
        # 买到本不该有的点燃资格（实测注入 μ=(0,2.0) 可越过 SALIENCE_THRESHOLD）。
        # 纯边界收紧：合法输入行为逐字不变（零回归）。NaN 亦由此条拦下。
        if not (-1.0 <= mu[0] <= 1.0 and -1.0 <= mu[1] <= 1.0):
            raise ExternalPriorError(
                f"external_priors[{i}] ({name!r}) 的 μ 须在 [-1, 1] 内，"
                f"实际 μv={mu[0]}, μa={mu[1]}；请检查 MCP ModalityPrior 输出"
            )

        # M2：生理流 valence 精度强制归 MIN_PRECISION（无条件·唯一失真必改）。
        # 置于 M3 校验之前（code-reviewer W1 2026-07-15）：physio 的 Πv 无条件覆写，
        # 不因 MCP 误传超 cap / 非正 Πv 而在 M3 处误报——覆写后 Πv=MIN_PRECISION∈(0,cap]。
        if name.lower().startswith(_PHYSIO_PREFIXES):
            pi_v = MIN_PRECISION
            # μv 一并归零（2026-07-29 跨仓议定·两侧各封一半）。此前 M2 只覆写 Πv、**从不碰 μ**，
            # 而配套项目把「physio 的 μv 恒 0」误归因给「Zero M2」并据此推出其自律上界 0.359
            # —— 一条**假的跨仓依据**比数值本身危险。
            # 为什么归零是 M2 的自洽性补完而非新增约束：M2 的宪章是「生理对效价盲」，
            # 今天只对 Πv 兑现、对 μv 不兑现；而 salience = hypot(μ)·mean(Π) ⇒ 非零 μv 能在
            # **不换取任何后验影响力**的前提下（Πv=MIN_PRECISION 已把效价贡献压到可忽略）
            # 单买点燃资格——与越界 μ「买到本不该有的点燃资格」是同一失效模式换了个入口。
            # 数值后果：不归零时 hypot(μ) 最大到 √2，真实自点燃上界从 0.359 收紧到
            # `2*SALIENCE_THRESHOLD/√2 − MIN_PRECISION`（**现算，禁止手抄常量**）。
            # ⚠ 归零发生在 M7 之后：越域 μv 仍先被 M7 拒绝，不是「静默接受再抹掉」。
            mu = (0.0, mu[1])

        # M3′：精度有限性校验（NaN/±inf）。**必须先于**下面两条比较——NaN 与任何数比较恒 False，
        # 故 `pi_v <= 0.0` 与 `pi_v > precision_cap` 对 NaN **双双放行**，NaN 精度会一路进
        # fuse_terms 产出 NaN 后验且全程无报错（±inf 反而会被下面两条接住，真正的洞只有 NaN）。
        # μ 侧不需要本条：M7 的 `-1.0 <= mu <= 1.0` 对 NaN 恒 False，上面已 raise。
        # 置于 M2 之后（同 M3 的理由）：physio 的 Πv 已被无条件覆写为 MIN_PRECISION，
        # 不因 MCP 误传 NaN Πv 而在此误报。
        # 跨仓背景：Zero_MCP 已在其 client 侧单边兜住（`479b887`），但那保护不了 chat 面与
        # 测试夹具——`mapping.py` 的 `float(prec[0])` 会把字符串 "nan" 原样转成 NaN 送进来。
        # 回归锁：tests/test_external_priors.py::TestExpandExternalPriorsM3 的 nan/inf 四例
        # （撤掉本块即红），另附 `test_old_comparisons_would_have_passed_nan` 自证判别力。
        if not (math.isfinite(pi_v) and math.isfinite(pi_a)):
            raise ExternalPriorError(
                f"external_priors[{i}] ({name!r}) 精度须为有限数，"
                f"实际 Πv={pi_v}, Πa={pi_a}；请检查 MCP 传参"
            )

        # M3：精度正值校验（physio Πv 已被 M2 覆写为 MIN_PRECISION>0；Πa 恒校验）
        if pi_v <= 0.0 or pi_a <= 0.0:
            raise ExternalPriorError(
                f"external_priors[{i}] ({name!r}) 精度须 >0，"
                f"实际 Πv={pi_v}, Πa={pi_a}；请检查 MCP 传参"
            )
        # M3：精度上界校验（physio Πv=MIN_PRECISION<<cap 必过；Πa 恒校验）
        if pi_v > precision_cap or pi_a > precision_cap:
            raise ExternalPriorError(
                f"external_priors[{i}] ({name!r}) 精度超过上界 {precision_cap}，"
                f"实际 Πv={pi_v}, Πa={pi_a}；请降低 MCP 精度或调高 "
                f"ZERO_EXTERNAL_PRIOR_PRECISION_CAP"
            )

        result.append((name, mu, (pi_v, pi_a)))

    return result
