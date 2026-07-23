"""情感流水线的纯数学内核（可独立单测）。

贝叶斯流水线：OCC 先验 → TD/精度 → 高斯积融合 → 后验采样 → 通道解码。
含数值钳制：精度/方差下限、采样方差上界，保证不发散。
此模块为纯函数，无 I/O、无副作用。
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

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
    delta: float, value_estimate: float, *, alpha: float = 1.0, beta: float = 0.5
) -> float:
    """精度 π = σ(α·|δ| + β·V)：RPE 强度与价值确定性共同决定证据权重。"""
    return max(MIN_PRECISION, sigmoid(alpha * abs(delta) + beta * value_estimate))


def precision_da(delta: float, *, alpha: float = 1.0) -> float:
    """DA 路径精度 π_DA = σ(α·|δ|)：消去 β·V，仅由 RPE 幅度决定证据权重。

    议会裁决 A-P1-A（神经席 M5 + 数学席 M6）：`precision(δ, V)` 原式 σ(α|δ|+β·V) 把
    DA 精度与价值混同，β·V 无神经依据。此函数只保留 DA 通路真正编码的信号——预测误差
    幅度 |δ|；value_estimate 项移除。精度下界与 `precision` 保持一致（MIN_PRECISION 钳制）。
    纯函数、无 I/O、无 env（守热路径红线）。
    """
    return max(MIN_PRECISION, sigmoid(alpha * abs(delta)))


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
    """
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
        # 双层容量门独立治理；WARN-3 fear 专属门（fear_domain_enabled·state.py:291）治标签/符号层，
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
) -> tuple[tuple[float, float], tuple[float, float]]:
    """快生存流（快速皮层下/防御回路，仿 LeDoux 生存回路思路）：从原始特征出粗 (μ, Π)。

    亚符号、低精度、最快：只用目标一致性符号定效价方向、强度定唤醒，不做 OCC 多维细评
    （那是慢评价流的事）。features = [goal, standard, attitude, intensity]（PerceptionAgent）。
    精度固定为 SURVIVAL_PRECISION（粗快=不确定），逐维返回。纯函数、无副作用。
    注（议会 B-1/M2）：具体解剖通路（上丘→丘脑枕→杏仁核"捷径"）在人类证据仍有争议
    （Pessoa & Adolphs 2010, NRN 11:773；LeDoux & Brown 2017），此处仅作快速显著性评估的
    工程近似，不承诺特定解剖底物。
    """
    goal = features[0] if features else 0.0
    intensity = features[3] if len(features) > 3 else 1.0
    valence = clamp(0.6 * goal, -1.0, 1.0)  # 粗：只取目标符号
    arousal = clamp(0.5 + 0.5 * abs(intensity), 0.0, 1.0)  # 威胁/显著 → 高唤醒
    return (valence, arousal), (SURVIVAL_PRECISION, SURVIVAL_PRECISION)


def stream_salience(mu: tuple[float, float], precision: tuple[float, float]) -> float:
    """显著网络（前岛叶+dACC）的门控分数：精度加权的偏离中性幅度 |μ|·Π̄。

    偏离中性越远、精度越高 → 越显著、越该进入全局广播。纯函数。
    """
    deviation = math.hypot(mu[0], mu[1])
    mean_precision = 0.5 * (precision[0] + precision[1])
    return deviation * mean_precision


def ignite(
    streams: list[tuple[str, tuple[float, float], tuple[float, float]]],
    *,
    threshold: float = SALIENCE_THRESHOLD,
    survival_fallback: bool = False,
    soft_beta: float | None = IGNITION_BETA,
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

    streams = [(name, μ, Π), ...]；返回 (点燃流的 [(μ, Π)] 供 fuse_terms, 点燃流名列表)。
    纯函数、无副作用。
    """
    scored = [(name, mu, prec, stream_salience(mu, prec)) for name, mu, prec in streams]

    if soft_beta is not None:
        # 软门控分支：所有流参与融合，精度按 logistic gate 调制
        terms = []
        names = []
        for name, mu, prec, sal in scored:
            gate = sigmoid(soft_beta * (sal - threshold))
            pi_eff: tuple[float, float] = (prec[0] * gate, prec[1] * gate)
            terms.append((mu, pi_eff))
            names.append(name)
        return terms, names

    # 硬 step 分支（soft_beta=None）：逐字旧行为，零回归
    fired = [(name, mu, prec) for name, mu, prec, s in scored if s >= threshold]
    if not fired:
        if survival_fallback:
            surv = next(
                ((name, mu, prec) for name, mu, prec, _ in scored if name == "survival"),
                None,
            )
            if surv is not None:
                fired = [surv]
            else:
                top = max(scored, key=lambda item: item[3])
                fired = [(top[0], top[1], top[2])]
        else:
            top = max(scored, key=lambda item: item[3])
            fired = [(top[0], top[1], top[2])]
    terms = [(mu, prec) for _, mu, prec in fired]
    names = [name for name, _, _ in fired]
    return terms, names


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
# 无条件覆写 Πv=MIN_PRECISION（EDA/HRV/瞳孔/SCR 对效价盲，Kreibig 2010）。
_PHYSIO_PREFIXES: tuple[str, ...] = ("physio", "eda", "hrv", "pupil", "scr")


def expand_external_priors(
    external_priors: list[ExternalPrior],
    *,
    precision_cap: float,
    max_streams: int,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """外部多模态先验流展开 + 防御性校验，返回可直接 extend 进 streams 的列表。

    M6（数学席·流数上界）：len(external_priors) > max_streams → raise ValueError。
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
    依据：EDA/HRV/瞳孔只编码交感唤醒输出、对效价盲（Kreibig 2010）；
    给 valence 精度 = 主动注入偏差（design.md §二·生物席强制·收敛 a）。

    返回与 affect_core.py:77 streams 类型完全一致的列表，可直接 extend（M1）。
    不进 occ_prior/survival 入口（design.md 受约束方案 c）。

    引文：Kreibig S.D. (2010). Biol. Psychol. 84(3):394-421.
    https://doi.org/10.1016/j.biopsycho.2010.03.010
    设计决策见 design.md M1–M6（议会 2026-07-15）。
    """
    # M6：流数上界
    if len(external_priors) > max_streams:
        raise ValueError(
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
            raise ValueError(
                f"external_priors[{i}] 形状不合法：须为 (str, (float,float), (float,float))，"
                f"实际为 {prior!r}；请检查 MCP as_zero_streams() 输出格式"
            )
        name: str = prior[0]
        mu: tuple[float, float] = (float(prior[1][0]), float(prior[1][1]))
        pi_v: float = float(prior[2][0])
        pi_a: float = float(prior[2][1])

        # M2：生理流 valence 精度强制归 MIN_PRECISION（无条件·唯一失真必改）。
        # 置于 M3 校验之前（code-reviewer W1 2026-07-15）：physio 的 Πv 无条件覆写，
        # 不因 MCP 误传超 cap / 非正 Πv 而在 M3 处误报——覆写后 Πv=MIN_PRECISION∈(0,cap]。
        if name.lower().startswith(_PHYSIO_PREFIXES):
            pi_v = MIN_PRECISION

        # M3：精度正值校验（physio Πv 已被 M2 覆写为 MIN_PRECISION>0；Πa 恒校验）
        if pi_v <= 0.0 or pi_a <= 0.0:
            raise ValueError(
                f"external_priors[{i}] ({name!r}) 精度须 >0，"
                f"实际 Πv={pi_v}, Πa={pi_a}；请检查 MCP 传参"
            )
        # M3：精度上界校验（physio Πv=MIN_PRECISION<<cap 必过；Πa 恒校验）
        if pi_v > precision_cap or pi_a > precision_cap:
            raise ValueError(
                f"external_priors[{i}] ({name!r}) 精度超过上界 {precision_cap}，"
                f"实际 Πv={pi_v}, Πa={pi_a}；请降低 MCP 精度或调高 "
                f"ZERO_EXTERNAL_PRIOR_PRECISION_CAP"
            )

        result.append((name, mu, (pi_v, pi_a)))

    return result
