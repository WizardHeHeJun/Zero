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
