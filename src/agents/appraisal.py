"""AppraisalAgent：OCC 评价节点（理性先验），产出先验分布与 reward。

对应 mPFC/ACC + 杏仁核效价标记。产出的 reward = 目标一致性，供 ValueAgent
做 TD（闭合评价 2 ↔ 价值 3）。节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import occ_prior
from src.orchestration.state import AffectState


class AppraisalAgent:
    """规则化 OCC 评价 → valence-arousal 先验 N(mu, sigma) + reward。"""

    def __call__(self, state: AffectState) -> dict:
        stim = state.stimulus
        if stim is None:
            return {}
        prior_mu, prior_sigma, reward = occ_prior(
            stim.goal_congruence,
            stim.standard_compliance,
            stim.attitude_appeal,
            stim.intensity,
        )
        appraisal = {"valence": prior_mu[0], "arousal": prior_mu[1]}
        entry = {
            "node": "appraisal",
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "reward": reward,
        }
        return {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "reward": reward,
            "appraisal": appraisal,
            "trace": state.trace + [entry],
        }
