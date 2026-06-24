"""AppraisalAgent：OCC 评价节点（理性先验），产出先验分布与 reward。

对应 mPFC/ACC + 杏仁核效价标记。产出的 reward = 目标一致性，供 ValueAgent
做 TD（闭合评价 2 ↔ 价值 3）。若 `state.recalled_disposition` 存在（记忆读闭环），
按 RECALL_BIAS_WEIGHT 把长期情绪倾向叠加到先验 valence——当前评价被长期人格弯折。
reward 不受回灌影响（保持目标一致性），故 TD/价值通路不变。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import clamp, occ_prior
from src.orchestration.state import AffectState

RECALL_BIAS_WEIGHT = 0.3  # 长期倾向对先验 valence 的偏置强度


class AppraisalAgent:
    """规则化 OCC 评价 → valence-arousal 先验 N(mu, sigma) + reward，可选回灌偏置。"""

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
        if state.recalled_disposition is not None:
            biased_v = clamp(
                prior_mu[0] + RECALL_BIAS_WEIGHT * state.recalled_disposition, -1.0, 1.0
            )
            prior_mu = (biased_v, prior_mu[1])
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
            "trace": [entry],
        }
