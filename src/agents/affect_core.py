"""AffectCoreAgent：主动推断生成核（随机性来源）。

对应岛叶 + 内感受预测层级。把 OCC 先验与 RPE 驱动的证据按精度做高斯积融合，
再从后验**采样**出最终情绪 e*=(valence, arousal)——随机性在此进入。
`mood_enabled` 时把「已在的心境」并入为第三个精度加权先验（**读**）；心境的**更新**
由本节点之后独立的 MoodAgent 完成（A.7）。默认行为与 v1 一致。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

import logging
import random

from src.agents.affect_math import (
    AROUSAL_GAIN,
    MIN_SIGMA,
    MOOD_PRECISION,
    TEXT_AFFECT_PRECISION,
    evidence_from_value,
    fast_survival_prior,
    fuse_terms,
    gaussian_fuse,
    ignite,
    sample_affect,
)
from src.orchestration.state import AffectState

logger = logging.getLogger(__name__)


class AffectCoreAgent:
    """高斯积融合先验与证据 → 后验 → 采样 e*；可选并入慢变心境（读，A.7）。"""

    def __call__(self, state: AffectState) -> dict:
        if state.prior_mu is None or state.prior_sigma is None or state.reward is None:
            return {}
        delta = state.rpe if state.rpe is not None else 0.0
        pi = state.precision if state.precision is not None else 1.0
        evidence = evidence_from_value(state.reward, delta)
        ignited: list[str] = []
        if state.workspace_enabled:
            # 显著度门控全局工作空间（v3）：并行流 (name, μ, Π) 竞争 → ignition 广播。
            # NE/唤醒增益：唤醒越高，评价·价值流的精度（投票权）越大（精度=神经调质增益）。
            arousal_gain = 1.0 + AROUSAL_GAIN * max(0.0, state.prior_mu[1])
            prior_prec = (
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[0]) ** 2,
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[1]) ** 2,
            )
            surv_mu, surv_prec = fast_survival_prior(state.features)
            streams: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
                ("survival", surv_mu, surv_prec),  # 快生存流（低精度、可单独点燃）
                ("appraisal", state.prior_mu, prior_prec),  # 慢评价流（OCC）
                ("value", evidence, (pi * arousal_gain, pi * arousal_gain)),  # 价值流（RPE）
            ]
            if state.mood_enabled and state.mood is not None:
                streams.append(("mood", state.mood, (MOOD_PRECISION, MOOD_PRECISION)))
            if state.text_affect is not None:
                streams.append(
                    ("text", state.text_affect, (TEXT_AFFECT_PRECISION, TEXT_AFFECT_PRECISION))
                )
            terms, ignited = ignite(streams)
            post_mu, post_sigma = fuse_terms(terms)
        elif state.mood_enabled:
            # 把「已在的心境」作为第三个精度加权先验并入融合（当前被过去弯折）
            prev_mood = state.mood if state.mood is not None else (0.0, 0.0)
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
        e_star = sample_affect(post_mu, post_sigma, rng=rng, sigma_cap=state.sample_sigma_cap)
        entry: dict = {
            "node": "affect_core",
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
        }
        out: dict = {
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "affect_sample": e_star,
            "trace": [entry],
        }
        # 工作空间额外产出：点燃的流名 + 后验精度（供语言层精度加权再入）。
        # 默认关时返回 dict 与 trace 与 v1/v2 逐字一致（零回归）。
        if state.workspace_enabled:
            entry["ignited_streams"] = ignited
            out["ignited_streams"] = ignited
            out["affect_precision"] = 0.5 * (1.0 / post_sigma[0] ** 2 + 1.0 / post_sigma[1] ** 2)
        logger.debug(
            "affect_core e*=%s post_mu=%s post_sigma=%s ignited=%s",
            e_star,
            post_mu,
            post_sigma,
            ignited,
        )
        return out
