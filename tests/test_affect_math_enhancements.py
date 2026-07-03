"""B7a 四项数学增强的单测：零回归断言 + 开启功能断言。

覆盖：
  A-P2-A  occ_prior va_coupling 非对称（默认 0.6/0.6 = 旧行为）
  A-P2-E  attitude_step arousal_weight（默认 0.0 = 旧行为）
  A-P3-D  habituation_factor 双过程 sensitization（默认 gain=0.0 = 旧行为）
  A-P2-C  suppress_expression 新纯函数（Gross 1998 response-focused）
"""

from __future__ import annotations

import math

import pytest

from src.agents.affect_math import (
    ATTITUDE_RATE,
    ATTITUDE_REVERSION,
    attitude_step,
    habituation_factor,
    occ_prior,
    suppress_expression,
)

# ---------------------------------------------------------------------------
# A-P2-A  occ_prior va_coupling 非对称
# ---------------------------------------------------------------------------


class TestOccPriorVaCoupling:
    """va_coupling_pos/neg 默认均 0.6，等价于原 0.6*|valence|（零回归）。"""

    def _arousal_only(
        self,
        goal: float,
        standard: float,
        attitude: float,
        intensity: float,
        **kw: float,
    ) -> float:
        """仅取 arousal 输出。"""
        _, (mu_v, mu_a), _ = occ_prior(goal, standard, attitude, intensity, **kw)  # type: ignore[misc]
        (mu_v2, mu_a2), _, _ = occ_prior(goal, standard, attitude, intensity, **kw)
        return mu_a2

    def test_zero_regression_default_equals_abs_valence_formula(self) -> None:
        """默认参数下 arousal 与手算 0.6*|valence| 逐字相等。"""
        cases = [
            (1.0, 0.5, 0.3, 0.8),  # 正效价
            (-1.0, -0.5, -0.3, 0.8),  # 负效价
            (0.0, 0.0, 0.0, 0.5),  # 中性
        ]
        for goal, std, att, inten in cases:
            (mu_v, mu_a), (sig_v, sig_a), reward = occ_prior(goal, std, att, inten)
            # 手算旧公式：valence = clamp(0.5*goal + 0.3*std + 0.2*att, -1, 1)
            valence_ref = max(-1.0, min(1.0, 0.5 * goal + 0.3 * std + 0.2 * att))
            arousal_ref = max(-1.0, min(1.0, 0.4 * abs(inten) + 0.6 * abs(valence_ref)))
            assert mu_a == pytest.approx(arousal_ref, abs=1e-9), (
                f"默认参数 arousal 应与旧公式相等 (case={goal},{std},{att},{inten})"
            )

    def test_zero_regression_explicit_0_6_equals_default(self) -> None:
        """显式传 0.6/0.6 与默认参数数值完全相等。"""
        args = (0.6, -0.4, 0.2, 0.7)
        default = occ_prior(*args)
        explicit = occ_prior(*args, va_coupling_pos=0.6, va_coupling_neg=0.6)
        assert default == explicit

    def test_neg_coupling_higher_raises_arousal_for_negative_valence(self) -> None:
        """负效价侧 coupling 更高（neg>pos）→ 负效价输入 arousal 更大（negativity bias）。"""
        args = (-1.0, -0.5, -0.3, 0.5)  # 使 valence < 0
        (_, arousal_default), _, _ = occ_prior(*args)
        (_, arousal_neg_bias), _, _ = occ_prior(*args, va_coupling_pos=0.5, va_coupling_neg=0.7)
        assert arousal_neg_bias > arousal_default, "负效价侧 coupling 提高应使 arousal 更大"

    def test_pos_coupling_higher_raises_arousal_for_positive_valence(self) -> None:
        """正效价侧 coupling 更高 → 正效价输入 arousal 更大。"""
        args = (1.0, 0.5, 0.3, 0.5)  # 使 valence > 0
        (_, arousal_default), _, _ = occ_prior(*args)
        (_, arousal_pos_bias), _, _ = occ_prior(*args, va_coupling_pos=0.8, va_coupling_neg=0.4)
        assert arousal_pos_bias > arousal_default

    def test_asymmetry_positive_valence_unaffected_by_neg_coupling(self) -> None:
        """正效价时改变 va_coupling_neg 不影响结果（max(-valence,0)=0）。"""
        args = (1.0, 0.5, 0.0, 0.5)  # valence > 0
        result_a = occ_prior(*args, va_coupling_neg=0.1)
        result_b = occ_prior(*args, va_coupling_neg=0.9)
        assert result_a == result_b, "正效价下 neg coupling 应无影响"


# ---------------------------------------------------------------------------
# A-P2-E  attitude_step arousal_weight
# ---------------------------------------------------------------------------


class TestAttitudeStepArousalWeight:
    """arousal_weight=0.0（默认）= 旧行为；>0 使高唤醒 stimulus 加快态度形成。"""

    def test_zero_regression_default_equals_no_weight(self) -> None:
        """默认 arousal_weight=0.0 结果与显式 0.0 完全相等。"""
        att = (0.2, -0.1)
        stim = (0.5, 0.8)
        default = attitude_step(att, stim)
        explicit = attitude_step(att, stim, arousal_weight=0.0)
        assert default == explicit

    def test_zero_regression_rate_eff_equals_rate_when_weight_zero(self) -> None:
        """weight=0 时 rate_eff=rate，与旧 attitude_step 数值逐字相等。"""
        att = (0.0, 0.0)
        stim = (0.6, 0.4)
        result_new = attitude_step(att, stim, arousal_weight=0.0)
        # 手算旧公式（reversion=ATTITUDE_REVERSION=0.01）
        rate = ATTITUDE_RATE
        rev = ATTITUDE_REVERSION
        v_ref = max(-1.0, min(1.0, (1.0 - rate) * 0.0 + rate * 0.6 - rev * (0.0 - 0.0)))
        a_ref = max(-1.0, min(1.0, (1.0 - rate) * 0.0 + rate * 0.4 - rev * (0.0 - 0.0)))
        assert result_new[0] == pytest.approx(v_ref, abs=1e-12)
        assert result_new[1] == pytest.approx(a_ref, abs=1e-12)

    def test_arousal_weight_accelerates_attitude_formation(self) -> None:
        """arousal_weight>0 + 高唤醒 stimulus → 态度更快收敛（多轮后离 stimulus 更近）。"""
        att0 = (0.0, 0.0)
        stim = (0.8, 0.9)  # 高唤醒
        att_default = att0
        att_weighted = att0
        rounds = 30
        for _ in range(rounds):
            att_default = attitude_step(att_default, stim, arousal_weight=0.0)
            att_weighted = attitude_step(att_weighted, stim, arousal_weight=1.0)
        # arousal_weight>0 使有效累积率更高 → valence 维更快靠近 stimulus[0]=0.8
        assert att_weighted[0] > att_default[0], "高唤醒权重应加快 attitude valence 收敛"

    def test_arousal_weight_zero_arousal_stimulus_no_effect(self) -> None:
        """stimulus arousal=0 时 arousal_weight 无论多大都与默认相等（rate_eff=rate*1）。"""
        att = (0.3, 0.1)
        stim = (0.5, 0.0)  # arousal=0
        default = attitude_step(att, stim, arousal_weight=0.0)
        weighted = attitude_step(att, stim, arousal_weight=5.0)
        assert default == weighted


# ---------------------------------------------------------------------------
# A-P3-D  habituation_factor 双过程
# ---------------------------------------------------------------------------


class TestHabituationFactorDualProcess:
    """sensitization_gain=0.0（默认）= 旧行为 exp(−n/τ)；>0 允许强刺激 η>1。"""

    def test_zero_regression_default_equals_pure_exp(self) -> None:
        """默认参数 = 旧公式 exp(−n/τ)，不含敏化。"""
        for n, tau in [(0, 5.0), (3, 5.0), (10, 5.0), (1, 1.0)]:
            result = habituation_factor(n, tau)
            expected = math.exp(-n / tau)
            assert result == pytest.approx(expected, abs=1e-12), (
                f"默认参数应与旧公式相等 (n={n}, tau={tau})"
            )

    def test_zero_regression_tau_le_zero_returns_one(self) -> None:
        """tau<=0 仍返回 1.0（旧行为，零回归）。"""
        assert habituation_factor(5, 0.0) == 1.0
        assert habituation_factor(5, -1.0) == 1.0
        # 即使有 sensitization_gain，tau<=0 也返回 1.0
        assert habituation_factor(5, 0.0, sensitization_gain=1.0, intensity=0.9) == 1.0

    def test_zero_regression_explicit_zero_gain_equals_default(self) -> None:
        """显式 sensitization_gain=0.0 与默认参数结果完全相等。"""
        for n, tau in [(2, 8.0), (7, 5.0)]:
            assert habituation_factor(n, tau) == habituation_factor(n, tau, sensitization_gain=0.0)

    def test_sensitization_gain_raises_eta_above_habituation(self) -> None:
        """sensitization_gain>0 + 高强度 → η > 纯习惯化（敏化项提升响应）。"""
        n, tau = 5, 5.0
        eta_hab = habituation_factor(n, tau)
        eta_dual = habituation_factor(n, tau, intensity=0.9, sensitization_gain=0.5)
        assert eta_dual > eta_hab, "敏化增益应使 η 高于纯习惯化"

    def test_sensitization_gain_strong_stimulus_can_exceed_one(self) -> None:
        """强刺激 + 高增益 → η>1（敏化主导，超过基线响应）。"""
        eta = habituation_factor(1, 20.0, intensity=1.0, sensitization_gain=2.0)
        assert eta > 1.0, "敏化主导时 η 应可超过 1.0"

    def test_sensitization_below_threshold_no_effect(self) -> None:
        """intensity 低于 sensitization_threshold → 敏化项为零，η = 纯习惯化。"""
        n, tau = 3, 5.0
        eta_hab = habituation_factor(n, tau)
        # intensity=0.3 < 默认 threshold=0.5 → max(0.3-0.5,0)=0 → 敏化为零
        eta_dual = habituation_factor(n, tau, intensity=0.3, sensitization_gain=5.0)
        assert eta_dual == pytest.approx(eta_hab, abs=1e-12)

    def test_negative_exposure_treated_as_zero(self) -> None:
        """负 exposure 按 0 处理（旧行为保持）。"""
        assert habituation_factor(-3, 5.0) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# A-P2-C  suppress_expression
# ---------------------------------------------------------------------------


class TestSuppressExpression:
    """Gross 1998 response-focused：factor 缩放输出幅度，不改内部体验。"""

    def test_factor_one_no_change(self) -> None:
        """factor=1.0 → 幅度不变（无抑制）。"""
        affect = (0.7, -0.5)
        result = suppress_expression(affect, factor=1.0)
        assert result[0] == pytest.approx(0.7, abs=1e-12)
        assert result[1] == pytest.approx(-0.5, abs=1e-12)

    def test_factor_zero_output_zero(self) -> None:
        """factor=0.0 → 完全抑制，输出归零。"""
        result = suppress_expression((0.8, -0.6), factor=0.0)
        assert result == (0.0, 0.0)

    def test_factor_half_scales_both_dims(self) -> None:
        """factor=0.5 → 两维均减半。"""
        affect = (0.6, 0.4)
        result = suppress_expression(affect, factor=0.5)
        assert result[0] == pytest.approx(0.3, abs=1e-9)
        assert result[1] == pytest.approx(0.2, abs=1e-9)

    def test_output_clamped_to_bounds(self) -> None:
        """结果钳制 [-1, 1]（factor>1 时防越界）。"""
        # factor=2.0 超比例——结果仍钳在 [-1,1]
        result = suppress_expression((0.9, 0.9), factor=2.0)
        assert -1.0 <= result[0] <= 1.0
        assert -1.0 <= result[1] <= 1.0

    def test_negative_valence_preserved_sign(self) -> None:
        """负效价经抑制后仍为负（符号保持，只压幅度）。"""
        affect = (-0.8, 0.3)
        result = suppress_expression(affect, factor=0.4)
        assert result[0] < 0.0
        assert result[0] == pytest.approx(-0.32, abs=1e-9)

    def test_distinct_from_reappraise(self) -> None:
        """与 reappraise 区分：抑制不改 valence 符号/方向，重评才改（抑制=压幅，重评=改义）。"""
        from src.agents.affect_math import reappraise

        affect = (-0.8, 0.7)
        suppressed = suppress_expression(affect, factor=0.5)
        reappraised = reappraise(affect)
        # 抑制：负效价仍为负
        assert suppressed[0] < 0.0
        # 重评：负效价被上抬（向 anchor 拉拢）
        assert reappraised[0] > affect[0]
