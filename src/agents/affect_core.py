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
    MIN_PRECISION,
    MIN_SIGMA,
    evidence_from_value,
    fast_survival_prior,
    fuse_terms,
    gaussian_fuse,
    hierarchical_fuse,
    ignite,
    precision_da,
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
            # P4-d（议会二轮·廉价 cap 防御）：默认 None → 不 cap（线性无界，零回归）；设 cap 则钳
            # arousal_gain≤1+cap，防高唤醒段 LC-NE 正反馈无界（完整倒 U 立项排后）。纯标量。
            if state.arousal_gain_cap is not None:
                arousal_gain = min(arousal_gain, 1.0 + state.arousal_gain_cap)
            prior_prec = (
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[0]) ** 2,
                arousal_gain / max(MIN_SIGMA, state.prior_sigma[1]) ** 2,
            )
            surv_mu, surv_prec = fast_survival_prior(state.features)

            # A-P1-A（议会裁决·神经席 M5+数学席 M6）：precision_split 门控。
            # True → value 流证据精度用 precision_da(rpe)（仅 DA 路径 |δ|，消 β·V 混同）；
            # False → 逐字旧行为 pi*arousal_gain（零回归）。
            rpe_for_da = state.rpe if state.rpe is not None else 0.0
            if state.precision_split:
                pi_da = precision_da(rpe_for_da) * arousal_gain
            else:
                pi_da = pi * arousal_gain

            # A-P0-B（议会裁决·数学席 M1）：fuse_independence_correct 门控。
            # value 流 valence 维与 appraisal/survival 共线（均依赖 goal_congruence）→
            # 条件独立假设失立 → 精度相加过度自信。
            # True → valence 维精度置 MIN_PRECISION（仅保留 arousal 维 π_DA，独立信号）；
            # False → 逐字旧行为（两维同精度，零回归）。
            # 评价流（appraisal）精度 = prior_prec，即 arousal_gain 加成后的 NE 路径 pi_ne。
            if state.fuse_independence_correct:
                value_prec: tuple[float, float] = (MIN_PRECISION, pi_da)  # valence 维极小
            else:
                value_prec = (pi_da, pi_da)

            streams: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
                ("survival", surv_mu, surv_prec),  # 快生存流（低精度、可单独点燃）
                (
                    "appraisal",
                    state.prior_mu,
                    prior_prec,
                ),  # 慢评价流（OCC）；精度=arousal_gain/σ²，即 NE 路径 pi_ne
                ("value", evidence, value_prec),  # 价值流（RPE）
            ]
            if state.mood_enabled and state.mood is not None:
                streams.append(("mood", state.mood, (state.mood_precision, state.mood_precision)))
            if state.text_affect is not None:
                streams.append(
                    (
                        "text",
                        state.text_affect,
                        (state.text_affect_precision, state.text_affect_precision),
                    )
                )
            terms, ignited = ignite(
                streams,
                survival_fallback=state.ignition_survival_fallback,
                soft_beta=state.ignition_beta,
            )
            # P3 层级预测编码（HPC v1）：门控关（默认 layers=1 或 coupling=0.0）→
            # 走现 fuse_terms 路径逐字不变（hierarchical_fuse 内部退化旁路保证零回归）。
            # 开启（layers>=2 且 coupling>0）→ 重建带 name 的流列表传给 hierarchical_fuse。
            # v1 在 ignite 之后做层级融合（保守取向；神经席「各流先层级、再进 GNW 竞争」的精确拓扑
            # 留 v2）。named_terms 用 ignite 输出的 terms（已含 soft_beta gate 调制后的精度，
            # soft_beta=None 时即原始精度）与 ignited 名单同序对齐重建——与软门控叠加时
            # 精度语义自洽（gate 先于 HPC）；soft_beta=None（默认）时精度不变，零回归。
            if state.hierarchical_layers >= 2 and state.hierarchical_coupling > 0.0:
                named_terms: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
                    (name, mu, prec) for name, (mu, prec) in zip(ignited, terms, strict=True)
                ]
                if named_terms:
                    post_mu, post_sigma = hierarchical_fuse(
                        named_terms,
                        layers=state.hierarchical_layers,
                        coupling=state.hierarchical_coupling,
                    )
                else:
                    post_mu, post_sigma = fuse_terms(terms)
            else:
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
                    (prev_mood, (state.mood_precision, state.mood_precision)),
                ]
            )
        else:
            post_mu, post_sigma = gaussian_fuse(state.prior_mu, state.prior_sigma, evidence, pi)
        # rng_seed 为空时每次调用都重新随机（有意：生产情绪表达的随机性），
        # 非漏传 seed；测试需可复现时显式传 rng_seed。
        rng = random.Random(state.rng_seed) if state.rng_seed is not None else None
        # P4（议会 α，数学+神经一致）：map 读出取后验均值 e*=post_mu（MMSE 最优点估计，消单样本大
        # 方差致的逐轮翻号，时序连续性交既有 emotion_decay_step 的 AR1≈0.4 承担）；默认 sample=逐轮
        # 后验采样（逐字旧行为，零回归）。sample_affect 保留——可供表达层做简并可变性。
        if state.affect_readout == "map":
            e_star = post_mu
        else:
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
