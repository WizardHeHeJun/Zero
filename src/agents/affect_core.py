"""AffectCoreAgent：主动推断生成核（随机性来源）。

对应岛叶 + 内感受预测层级。把 OCC 先验与 RPE 驱动的证据按精度做高斯积融合，
再从后验**采样**出最终情绪 e*=(valence, arousal)——随机性在此进入。
`mood_enabled` 时把「已在的心境」并入为第三个精度加权先验，并在采样后用 e* 更新
心境（A.7 时间深度/滞后）；默认关闭，行为与 v1 完全一致。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import random

from src.agents.affect_math import (
    MIN_SIGMA,
    MOOD_PRECISION,
    evidence_from_value,
    fuse_terms,
    gaussian_fuse,
    mood_step,
    sample_affect,
)
from src.orchestration.state import AffectState


class AffectCoreAgent:
    """高斯积融合先验与证据 → 后验 → 采样 e*；可选并入慢变心境（A.7）。"""

    def __call__(self, state: AffectState) -> dict:
        if state.prior_mu is None or state.prior_sigma is None or state.reward is None:
            return {}
        delta = state.rpe if state.rpe is not None else 0.0
        pi = state.precision if state.precision is not None else 1.0
        evidence = evidence_from_value(state.reward, delta)
        prev_mood = state.mood if state.mood is not None else (0.0, 0.0)
        if state.mood_enabled:
            # 把「已在的心境」作为第三个精度加权先验并入融合（当前被过去弯折）
            prior_prec = (
                1.0 / max(MIN_SIGMA, state.prior_sigma[0]) ** 2,
                1.0 / max(MIN_SIGMA, state.prior_sigma[1]) ** 2,
            )
            post_mu, post_sigma = fuse_terms(
                [
                    (state.prior_mu, prior_prec),
                    (evidence, (pi, pi)),
                    (prev_mood, (MOOD_PRECISION, MOOD_PRECISION)),
                ]
            )
        else:
            post_mu, post_sigma = gaussian_fuse(state.prior_mu, state.prior_sigma, evidence, pi)
        # rng_seed 为空时每次调用都重新随机（有意：生产情绪表达的随机性），
        # 非漏传 seed；测试需可复现时显式传 rng_seed。
        rng = random.Random(state.rng_seed) if state.rng_seed is not None else None
        e_star = sample_affect(post_mu, post_sigma, rng=rng)
        entry = {
            "node": "affect_core",
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
        }
        out = {
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
            "trace": [entry],
        }
        if state.mood_enabled:
            # 采样后用 e* 驱动心境的双稳更新；mood 作为运行态随 Checkpointer 持久化
            new_mood = mood_step(prev_mood, e_star)
            entry["mood"] = new_mood
            out["mood"] = new_mood
        return out
