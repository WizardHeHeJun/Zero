"""阶段16：attitude_appeal 进 AppraisalAgent 先验的测试。

覆盖：
1. occ_prior 方向性：attitude_appeal 正/负 → prior valence 正向单调（0.2 权重生效）。
2. 端到端 AppraisalAgent：attitude_appeal 高/低 → prior_mu valence 单调。
3. 两通路独立（防 double-counting）：attitude_appeal 走 occ_prior 内部、
   recalled_disposition 走输出后加法偏置，两者数值可分离、不重复计入。
4. 零回归：recalled_disposition=None 时行为与已有测试一致（reward 不受回灌影响）。
"""

from __future__ import annotations

import pytest

from src.agents.affect_math import clamp, occ_prior
from src.agents.appraisal import RECALL_BIAS_WEIGHT, AppraisalAgent
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 1. occ_prior 方向性：attitude_appeal 权重 0.2 正向单调
# ---------------------------------------------------------------------------


def test_occ_prior_attitude_appeal_positive_raises_valence() -> None:
    """attitude_appeal=+0.8 相比 -0.8，prior valence 更高（0.2 权重方向正确）。"""
    (v_pos, _), _, _ = occ_prior(
        goal_congruence=0.0,
        standard_compliance=0.0,
        attitude_appeal=0.8,
        intensity=0.5,
    )
    (v_neg, _), _, _ = occ_prior(
        goal_congruence=0.0,
        standard_compliance=0.0,
        attitude_appeal=-0.8,
        intensity=0.5,
    )
    assert v_pos > v_neg, f"attitude_appeal=+0.8 应产生更高 valence，got {v_pos=} vs {v_neg=}"


def test_occ_prior_attitude_appeal_monotonic_across_range() -> None:
    """attitude_appeal 从 -1 到 +1 逐步递增时，prior valence 单调不降。"""
    appeals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    valences = [
        occ_prior(
            goal_congruence=0.0,
            standard_compliance=0.0,
            attitude_appeal=a,
            intensity=0.5,
        )[0][0]
        for a in appeals
    ]
    for i in range(len(valences) - 1):
        assert valences[i] <= valences[i + 1], (
            f"valence 应单调不降：appeals[{i}]={appeals[i]} → {valences[i]}, "
            f"appeals[{i + 1}]={appeals[i + 1]} → {valences[i + 1]}"
        )


def test_occ_prior_attitude_appeal_weight_is_0_2() -> None:
    """attitude_appeal 差值 1.6（+0.8 vs -0.8）在 valence 中体现为 0.2*1.6=0.32。"""
    (v_pos, _), _, _ = occ_prior(0.0, 0.0, 0.8, 0.5)
    (v_neg, _), _, _ = occ_prior(0.0, 0.0, -0.8, 0.5)
    diff = v_pos - v_neg
    expected_diff = 0.2 * 1.6  # 0.32
    assert abs(diff - expected_diff) < 1e-9, f"valence 差应为 {expected_diff}，实际 {diff}"


def test_occ_prior_attitude_appeal_does_not_affect_reward() -> None:
    """attitude_appeal 变化不应影响 reward（reward=goal_congruence 钳制）。"""
    _, _, r_pos = occ_prior(0.5, 0.0, 0.8, 0.5)
    _, _, r_neg = occ_prior(0.5, 0.0, -0.8, 0.5)
    assert r_pos == r_neg == pytest.approx(0.5), (
        f"reward 应等于 goal_congruence，与 attitude_appeal 无关：{r_pos=}, {r_neg=}"
    )


# ---------------------------------------------------------------------------
# 2. 端到端 AppraisalAgent：attitude_appeal 高/低 → prior_mu valence 单调
# ---------------------------------------------------------------------------


def _make_state(attitude_appeal: float, recalled_disposition: float | None = None) -> AffectState:
    """辅助：构造带 attitude_appeal 的 AffectState。"""
    stim = Stimulus(
        name="test",
        goal_congruence=0.0,
        standard_compliance=0.0,
        attitude_appeal=attitude_appeal,
        intensity=0.5,
    )
    return AffectState(stimulus=stim, recalled_disposition=recalled_disposition)


def test_appraisal_agent_positive_attitude_appeal_raises_prior_valence() -> None:
    """attitude_appeal=+0.8 相比 -0.8，AppraisalAgent 产出 prior_mu valence 更高。"""
    agent = AppraisalAgent()
    out_pos = agent(_make_state(0.8))
    out_neg = agent(_make_state(-0.8))
    assert out_pos["prior_mu"][0] > out_neg["prior_mu"][0], (
        f"正态度应产出更高 prior valence：{out_pos['prior_mu'][0]=} vs {out_neg['prior_mu'][0]=}"
    )


def test_appraisal_agent_attitude_appeal_does_not_affect_reward() -> None:
    """attitude_appeal 不影响 AppraisalAgent 的 reward 输出。"""
    agent = AppraisalAgent()
    out_pos = agent(_make_state(0.8))
    out_neg = agent(_make_state(-0.8))
    assert out_pos["reward"] == out_neg["reward"], (
        f"reward 不应随 attitude_appeal 变化：{out_pos['reward']=} vs {out_neg['reward']=}"
    )


def test_appraisal_agent_returns_valid_increment_fields() -> None:
    """AppraisalAgent 只返回合法 AffectState 字段（节点契约）。"""
    agent = AppraisalAgent()
    out = agent(_make_state(0.5))
    legal_fields = set(AffectState.model_fields)
    assert set(out).issubset(legal_fields), f"非法字段：{set(out) - legal_fields}"
    assert "prior_mu" in out
    assert "prior_sigma" in out
    assert "reward" in out


# ---------------------------------------------------------------------------
# 3. 两通路独立：attitude_appeal（occ_prior 内部）vs recalled_disposition（输出后偏置）
# ---------------------------------------------------------------------------


def test_two_pathways_are_independent_no_double_counting() -> None:
    """验证 attitude_appeal 在 occ_prior 内、recalled_disposition 在输出后独立叠加。

    数学验证：
      occ_prior_valence = 0.5*gc + 0.3*sc + 0.2*aa
      final_valence = clamp(occ_prior_valence + RECALL_BIAS_WEIGHT * rd, -1, 1)

    两者对 final_valence 的贡献可精确分离。
    """
    agent = AppraisalAgent()
    gc, sc, _intensity = 0.0, 0.0, 0.5
    aa = 0.6
    rd = -0.5

    out = agent(_make_state(aa, recalled_disposition=rd))
    final_v = out["prior_mu"][0]

    # 手动重算两步
    expected_occ_v = 0.5 * gc + 0.3 * sc + 0.2 * aa  # = 0.12
    expected_final_v = clamp(expected_occ_v + RECALL_BIAS_WEIGHT * rd, -1.0, 1.0)

    assert abs(final_v - expected_final_v) < 1e-9, (
        f"两通路独立叠加计算不符：{final_v=} vs {expected_final_v=}"
    )


def test_attitude_appeal_contribution_separable_from_disposition() -> None:
    """attitude_appeal 对 prior valence 的贡献与 recalled_disposition 相互独立。

    具体：attitude_appeal 增量 Δaa → 先验 valence 增量 0.2*Δaa，
    recalled_disposition 增量 Δrd → 先验 valence 增量 RECALL_BIAS_WEIGHT*Δrd，
    两者线性叠加（无交叉项）。
    """
    agent = AppraisalAgent()

    # 基准
    base = agent(_make_state(0.0, recalled_disposition=0.0))
    base_v = base["prior_mu"][0]

    # 只变 attitude_appeal
    with_aa = agent(_make_state(0.4, recalled_disposition=0.0))
    delta_aa = with_aa["prior_mu"][0] - base_v
    expected_delta_aa = 0.2 * 0.4  # 0.08

    # 只变 recalled_disposition
    with_rd = agent(_make_state(0.0, recalled_disposition=0.4))
    delta_rd = with_rd["prior_mu"][0] - base_v
    expected_delta_rd = RECALL_BIAS_WEIGHT * 0.4  # 0.12

    # 两者同时变化
    with_both = agent(_make_state(0.4, recalled_disposition=0.4))
    delta_both = with_both["prior_mu"][0] - base_v

    assert abs(delta_aa - expected_delta_aa) < 1e-9, (
        f"attitude_appeal 贡献不符：{delta_aa=} vs {expected_delta_aa=}"
    )
    assert abs(delta_rd - expected_delta_rd) < 1e-9, (
        f"recalled_disposition 贡献不符：{delta_rd=} vs {expected_delta_rd=}"
    )
    # 两者同时等于各自单独贡献之和（可分离、无 double-counting）
    assert abs(delta_both - (expected_delta_aa + expected_delta_rd)) < 1e-9, (
        f"两通路叠加不满足可分离性：{delta_both=} vs {expected_delta_aa + expected_delta_rd=}"
    )


def test_recalled_disposition_bias_applied_after_occ_prior() -> None:
    """recalled_disposition 在 occ_prior 之后叠加，不进入 occ_prior 内部计算。

    验证：仅有 recalled_disposition 时，occ_prior 部分（gc=0, sc=0, aa=0）为 0.0，
    再叠加 disposition 偏置，与全零 occ_prior + bias 一致。
    """
    agent = AppraisalAgent()
    # recalled_disposition 走外部叠加，不混入 occ_prior
    out_with_rd = agent(_make_state(0.0, recalled_disposition=0.6))
    # 手动：occ_prior(0,0,0,0.5) valence = 0；再加 RECALL_BIAS_WEIGHT * 0.6
    raw_occ_v = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 0.0  # = 0.0
    expected_v = clamp(raw_occ_v + RECALL_BIAS_WEIGHT * 0.6, -1.0, 1.0)
    assert abs(out_with_rd["prior_mu"][0] - expected_v) < 1e-9, (
        f"disposition 应 occ_prior 后叠加：{out_with_rd['prior_mu'][0]=} vs {expected_v=}"
    )


# ---------------------------------------------------------------------------
# 4. 零回归：recalled_disposition=None 时行为与既有逻辑一致
# ---------------------------------------------------------------------------


def test_no_recalled_disposition_reward_equals_goal_congruence() -> None:
    """recalled_disposition=None 时，reward 就是 goal_congruence（钳制后）。"""
    agent = AppraisalAgent()
    gc = 0.7
    stim = Stimulus(name="z", goal_congruence=gc, intensity=0.5)
    out = agent(AffectState(stimulus=stim))
    assert out["reward"] == pytest.approx(clamp(gc, -1.0, 1.0))


def test_no_stimulus_returns_empty_dict() -> None:
    """stimulus=None 时，AppraisalAgent 返回空 dict（零回归：不 crash）。"""
    agent = AppraisalAgent()
    out = agent(AffectState())
    assert out == {}


def test_prior_valence_bounds_with_extreme_attitude_appeal() -> None:
    """attitude_appeal 极值时 prior_mu 仍在 [-1, 1]（钳制正常工作）。"""
    agent = AppraisalAgent()
    for aa in [-1.0, 1.0, -2.0, 2.0]:
        stim = Stimulus(name="ext", goal_congruence=0.9, intensity=1.0, attitude_appeal=aa)
        out = agent(AffectState(stimulus=stim))
        v, a = out["prior_mu"]
        assert -1.0 <= v <= 1.0, f"valence 越界：{v=} with attitude_appeal={aa}"
        assert -1.0 <= a <= 1.0, f"arousal 越界：{a=} with attitude_appeal={aa}"
