"""AffectCoreAgent：主动推断生成核（随机性来源）。

对应岛叶 + 内感受预测层级。把 OCC 先验与 RPE 驱动的证据按精度做高斯积融合，
再从后验**采样**出最终情绪 e*=(valence, arousal)——随机性在此进入。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import random

from src.agents.affect_math import evidence_from_value, gaussian_fuse, sample_affect
from src.orchestration.state import AffectState


class AffectCoreAgent:
    """高斯积融合先验与证据 → 后验 → 采样 e*。"""

    def __call__(self, state: AffectState) -> dict:
        if state.prior_mu is None or state.prior_sigma is None or state.reward is None:
            return {}
        delta = state.rpe if state.rpe is not None else 0.0
        pi = state.precision if state.precision is not None else 1.0
        evidence = evidence_from_value(state.reward, delta)
        post_mu, post_sigma = gaussian_fuse(state.prior_mu, state.prior_sigma, evidence, pi)
        rng = random.Random(state.rng_seed) if state.rng_seed is not None else None
        e_star = sample_affect(post_mu, post_sigma, rng=rng)
        entry = {
            "node": "affect_core",
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
        }
        return {
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
            "trace": state.trace + [entry],
        }
