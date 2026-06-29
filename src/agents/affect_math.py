"""情感流水线的纯数学内核（可独立单测）。

贝叶斯流水线：OCC 先验 → TD/精度 → 高斯积融合 → 后验采样 → 通道解码。
含数值钳制：精度/方差下限、采样方差上界，保证不发散。
此模块为纯函数，无 I/O、无副作用。
"""

from __future__ import annotations

import math
import random
from typing import Any

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
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """OCC 评价 → (prior_mu, prior_sigma, reward)。

    valence 由目标/标准/态度一致性线性合成；arousal 由强度与 |效价| 合成；
    sigma 随显著度上升而下降（越显著越确定）；reward = 目标一致性（闭合 2↔3）。
    """
    valence = clamp(
        0.5 * goal_congruence + 0.3 * standard_compliance + 0.2 * attitude_appeal,
        -1.0,
        1.0,
    )
    arousal = clamp(0.4 * abs(intensity) + 0.6 * abs(valence), -1.0, 1.0)
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
) -> tuple[float, float]:
    """从后验高斯采样 e*=(valence, arousal)；方差有界、结果钳制到 [-1, 1]。"""
    generator = rng if rng is not None else random.Random()
    v = clamp(generator.gauss(post_mu[0], min(MAX_SAMPLE_SIGMA, post_sigma[0])), -1.0, 1.0)
    a = clamp(generator.gauss(post_mu[1], min(MAX_SAMPLE_SIGMA, post_sigma[1])), -1.0, 1.0)
    return (v, a)


def text_label(valence: float, arousal: float) -> str:
    """按 valence-arousal 象限映射离散情绪词（占位与模型路径共用）。"""
    if valence >= 0:
        return "excited" if arousal >= 0.33 else "content"
    return "angry" if arousal >= 0.33 else "sad"


def decode_channels(affect: tuple[float, float]) -> dict[str, Any]:
    """把 (valence, arousal) 占位解码为 4 个表达通道的结构化结果。

    通道：FACS AU 向量 / 文本情绪标签 / 生理信号模拟 / 语音韵律参数。
    """
    valence, arousal = affect

    # 1) FACS AU：正效价→AU12/AU6（拉嘴角/抬脸颊），负效价→AU15/AU4（压嘴角/皱眉）
    facs_au: dict[str, float] = {}
    if valence >= 0:
        facs_au["AU12"] = clamp(valence, 0.0, 1.0)
        facs_au["AU06"] = clamp(0.6 * valence, 0.0, 1.0)
    else:
        facs_au["AU15"] = clamp(-valence, 0.0, 1.0)
        facs_au["AU04"] = clamp(-0.6 * valence, 0.0, 1.0)
    facs_au["intensity"] = clamp(abs(arousal), 0.0, 1.0)

    # 2) 文本情绪标签：按 valence-arousal 象限映射离散词
    label = text_label(valence, arousal)

    # 3) 生理信号模拟：arousal 驱动交感输出
    physiology = {
        "heart_rate_bpm": 70.0 + 40.0 * clamp(arousal, 0.0, 1.0),
        "skin_conductance": clamp(arousal, 0.0, 1.0),
        "pupil_mm": 3.0 + 2.0 * clamp(arousal, 0.0, 1.0),
    }

    # 4) 语音韵律：arousal→语速/能量，valence→音高基线
    prosody = {
        "speech_rate": 1.0 + 0.5 * clamp(arousal, -1.0, 1.0),
        "pitch": 1.0 + 0.3 * valence,
        "energy": clamp(0.5 + 0.5 * arousal, 0.0, 1.0),
    }

    return {
        "facs_au": facs_au,
        "text_label": label,
        "physiology": physiology,
        "prosody": prosody,
    }


def fast_survival_prior(
    features: list[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """快生存流（上丘-丘脑枕-杏仁核捷径 / LeDoux 生存回路）：从原始特征出粗 (μ, Π)。

    亚符号、低精度、最快：只用目标一致性符号定效价方向、强度定唤醒，不做 OCC 多维细评
    （那是慢评价流的事）。features = [goal, standard, attitude, intensity]（PerceptionAgent）。
    精度固定为 SURVIVAL_PRECISION（粗快=不确定），逐维返回。纯函数、无副作用。
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
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str]]:
    """GNW ignition：salience ≥ threshold 的流点燃进入全局广播，亚阈流停留局部。

    若无流过阈（全弱刺激）则保留 salience 最高者，保证总有输出（不空播）。
    streams = [(name, μ, Π), ...]；返回 (点燃流的 [(μ, Π)] 供 fuse_terms, 点燃流名列表)。
    纯函数、无副作用。
    """
    scored = [(name, mu, prec, stream_salience(mu, prec)) for name, mu, prec in streams]
    fired = [(name, mu, prec) for name, mu, prec, s in scored if s >= threshold]
    if not fired:
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
) -> tuple[float, float]:
    """慢变态度/印象：按 stimulus 缓慢累积 + 向个体基线弱回归（evaluative conditioning）。

    `a' = (1-rate)·a + rate·stimulus − reversion·(a − setpoint)`。rate 小（多轮才成形）→ 长期、
    稳定、对象指向的评价（Scherer/Frijda 的 sentiment/attitude 层）。`reversion` 项是议会必改：
    无它则持续同向 stimulus 把 attitude 单调推到极端（affective homeostasis 缺失 / 慢性应激无负
    反馈，Russell 2003 · Kuppens 2010）；有它则恒定刺激下稳态 a*≈rate·s/(rate+reversion)、被钳在
    |s| 内不无限漂移，刺激停歇时缓慢回基线。`reversion=0` 退化为旧纯 EWMA（零回归开关）。纯函数。
    """
    return (
        clamp(
            (1.0 - rate) * attitude[0]
            + rate * stimulus[0]
            - reversion * (attitude[0] - setpoint[0]),
            -1.0,
            1.0,
        ),
        clamp(
            (1.0 - rate) * attitude[1]
            + rate * stimulus[1]
            - reversion * (attitude[1] - setpoint[1]),
            -1.0,
            1.0,
        ),
    )


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
