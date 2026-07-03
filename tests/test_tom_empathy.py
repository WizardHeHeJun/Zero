"""P3 1-C ToM / 社会情绪：共情偏置正式单测（tests/test_tom_empathy.py）。

覆盖范围：
  1. 退化零回归：interlocutor_affect=None 或各 w=0 → prior_mu 逐字不变
  2. 传染差分有界：w_c=1 全同步、w_c=0 不变、∈[-1,1]、双维独立验证
  3. 开环：改 state 自身字段（mood/affect_sample）不改 interlocutor_affect
  4. CARE 触发：v_i<0 → CARE 偏置>0；v_i>0 → 不触发；v_i=0 边界不触发
  5. 替代喜悦触发：(v_i>threshold且a_i>0) → valence 上移；其他情况不触发
  6. CARE / 替代喜悦互斥（v<0 走 CARE、v>threshold 走替代）
  7. 语义边界（WARN-5）：interlocutor_affect 独立于 stim.goal_congruence，两者独立作用
  8. 热路径无 LLM/torch：静态检查 appraisal.py 共情块只用 math（clamp/max/+/*）
  9. 贯通零回归：SessionConfig 默认 4 字段；env → build_chat_driver → session.config；to_state_flags

期望值均从方程/SEED_VAD_LEXICON 派生，不硬编码。
"""

from __future__ import annotations

import ast
import inspect
import math

import pytest

from src.agents.affect_math import clamp, occ_prior
from src.agents.appraisal import AppraisalAgent
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ─────────────────────────────────────────────────────────────────
# 工具：构造最小 state
# ─────────────────────────────────────────────────────────────────


def _state_with_stim(
    goal_congruence: float = 0.0,
    intensity: float = 0.5,
    interlocutor_affect: tuple[float, float] | None = None,
    contagion_alpha: float = 0.0,
    care_bias_alpha: float = 0.0,
    vicarious_alpha: float = 0.0,
    vicarious_threshold: float = 0.3,
    **kwargs: object,
) -> AffectState:
    """构造带 Stimulus 的最小 AffectState，方便参数化测试。"""
    stim = Stimulus(
        name="test",
        goal_congruence=goal_congruence,
        intensity=intensity,
    )
    return AffectState(
        stimulus=stim,
        interlocutor_affect=interlocutor_affect,
        contagion_alpha=contagion_alpha,
        care_bias_alpha=care_bias_alpha,
        vicarious_alpha=vicarious_alpha,
        vicarious_threshold=vicarious_threshold,
        **kwargs,
    )


def _baseline_prior_mu(
    goal_congruence: float = 0.0,
    intensity: float = 0.5,
) -> tuple[float, float]:
    """不带共情偏置时的 prior_mu（从 occ_prior 方程派生）。"""
    mu, _, _ = occ_prior(goal_congruence, 0.0, 0.0, intensity)
    return mu


# ─────────────────────────────────────────────────────────────────
# 1. 退化零回归
# ─────────────────────────────────────────────────────────────────


class TestZeroRegression:
    """interlocutor_affect=None 或各 w=0 → AppraisalAgent 输出 prior_mu 逐字不变。"""

    def test_none_interlocutor_prior_mu_unchanged(self) -> None:
        """interlocutor_affect=None（默认）→ prior_mu 与无 interlocutor 时逐字一致。"""
        gc = 0.4
        state_no_interlocutor = _state_with_stim(goal_congruence=gc, interlocutor_affect=None)
        state_explicit_none = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=None,
            contagion_alpha=0.2,  # alpha 有值但 interlocutor=None → 不触发
        )
        out_no = AppraisalAgent()(state_no_interlocutor)
        out_none = AppraisalAgent()(state_explicit_none)
        # 两者 prior_mu 应相同（None → 完全跳过共情块）
        assert out_no["prior_mu"] == pytest.approx(out_none["prior_mu"], abs=1e-10)

    def test_all_alphas_zero_interlocutor_present_no_bias(self) -> None:
        """contagion=care=vicarious=0 且 interlocutor_affect 有值 → prior_mu 不改变。

        方程：各 alpha=0 → 传染 Δ=0、CARE_bias=0、vic_bias=0 → prior_mu 不变。
        """
        gc = 0.5
        state_baseline = _state_with_stim(goal_congruence=gc)
        state_with_interlocutor = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(-0.8, 0.6),  # 对方负面情绪
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.0,
        )
        out_baseline = AppraisalAgent()(state_baseline)
        out_with = AppraisalAgent()(state_with_interlocutor)
        assert out_baseline["prior_mu"] == pytest.approx(out_with["prior_mu"], abs=1e-10)

    def test_zero_regression_various_stim_inputs(self) -> None:
        """多种 goal_congruence 下，各 alpha=0 → prior_mu 逐字零回归。"""
        for gc in (-0.8, -0.3, 0.0, 0.3, 0.8):
            state_no = _state_with_stim(goal_congruence=gc)
            state_with = _state_with_stim(
                goal_congruence=gc,
                interlocutor_affect=(0.5, 0.5),
                contagion_alpha=0.0,
                care_bias_alpha=0.0,
                vicarious_alpha=0.0,
            )
            out_no = AppraisalAgent()(state_no)
            out_with = AppraisalAgent()(state_with)
            assert out_no["prior_mu"] == pytest.approx(out_with["prior_mu"], abs=1e-10), (
                f"gc={gc} 时 alpha=0 应零回归"
            )


# ─────────────────────────────────────────────────────────────────
# 2. 传染差分有界
# ─────────────────────────────────────────────────────────────────


class TestContagionDifferential:
    """传染差分：w_c=1 全同步、w_c=0 不变、∈[-1,1]、双维独立验证。"""

    def test_contagion_w1_full_sync_valence(self) -> None:
        """w_c=1 → prior_mu valence 完全变为 v_i（全同步）。

        方程：v0 += 1.0*(v_i - v0) = v_i。然后经 CARE/替代路径（不触发：v_i>0）→ clamp。
        """
        v_i, a_i = 0.7, 0.3  # 对方情绪（v_i>0 → 不触 CARE，不触替代：需 v_i>threshold）
        state = _state_with_stim(
            goal_congruence=0.0,
            intensity=0.5,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=1.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.0,
            vicarious_threshold=0.3,
        )
        # baseline prior_mu（w_c 传染前）
        mu_base = _baseline_prior_mu(goal_congruence=0.0, intensity=0.5)
        out = AppraisalAgent()(state)
        # w_c=1 → v0 = v_i（传染后 = 对方值）→ clamp(-1,1) = v_i（在范围内）
        expected_v = clamp(v_i, -1.0, 1.0)  # v_i=0.7 不需 clamp
        expected_a = clamp(a_i, -1.0, 1.0)  # a_i=0.3 不需 clamp
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
            f"w_c=1 全同步：valence 应={expected_v}, 得 {out['prior_mu'][0]}"
        )
        assert out["prior_mu"][1] == pytest.approx(expected_a, abs=1e-9), (
            f"w_c=1 全同步：arousal 应={expected_a}, 得 {out['prior_mu'][1]}"
        )
        # 已确认与 baseline 不同（有了偏置）
        assert out["prior_mu"][0] != pytest.approx(mu_base[0], abs=1e-6), "w_c=1 应改变 valence"

    def test_contagion_w0_no_change(self) -> None:
        """w_c=0 → prior_mu 与无 interlocutor 逐字一致（不变）。"""
        gc = 0.3
        state_no = _state_with_stim(goal_congruence=gc)
        state_w0 = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(-0.5, 0.8),
            contagion_alpha=0.0,
        )
        out_no = AppraisalAgent()(state_no)
        out_w0 = AppraisalAgent()(state_w0)
        assert out_no["prior_mu"] == pytest.approx(out_w0["prior_mu"], abs=1e-10)

    def test_contagion_partial_convergence_valence(self) -> None:
        """w_c=0.2 → prior_mu valence 介于自身与对方之间（部分趋同）。

        方程：v0_new = v0 + 0.2*(v_i - v0) = 0.8*v0 + 0.2*v_i
        → v0_new ∈ (min(v0,v_i), max(v0,v_i))（凸组合，有界）。
        """
        gc = 0.0  # 中性刺激
        v_i, a_i = 0.8, 0.4  # 对方正面情绪（v_i>0 不触 CARE；v_i=0.8>threshold=0.3）
        w_c = 0.2
        # 先算不带 interlocutor 的 baseline
        state_no = _state_with_stim(goal_congruence=gc)
        mu_base = AppraisalAgent()(state_no)["prior_mu"]
        v0, a0 = mu_base

        # 传染方程（v_i>threshold 且无 vicarious_alpha → 只传染）
        state_w = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=w_c,
            care_bias_alpha=0.0,
            vicarious_alpha=0.0,  # vicarious 关闭
            vicarious_threshold=0.3,
        )
        out = AppraisalAgent()(state_w)
        # 期望：v0_new = v0 + 0.2*(v_i - v0)；需经 clamp
        expected_v = clamp(v0 + w_c * (v_i - v0), -1.0, 1.0)
        expected_a = clamp(a0 + w_c * (a_i - a0), -1.0, 1.0)
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
            f"w_c=0.2 部分趋同：valence 期望={expected_v:.4f}, 得={out['prior_mu'][0]:.4f}"
        )
        assert out["prior_mu"][1] == pytest.approx(expected_a, abs=1e-9), (
            f"w_c=0.2 部分趋同：arousal 期望={expected_a:.4f}, 得={out['prior_mu'][1]:.4f}"
        )
        # 介于两者之间（凸组合保证）
        v_min, v_max = min(v0, v_i), max(v0, v_i)
        assert (
            v_min <= out["prior_mu"][0] <= v_max
            or math.isclose(out["prior_mu"][0], v_min, abs_tol=1e-9)
            or math.isclose(out["prior_mu"][0], v_max, abs_tol=1e-9)
        ), "w_c=0.2 → 部分趋同值应在 v0 和 v_i 之间"

    def test_contagion_result_clamped_in_minus1_1(self) -> None:
        """传染结果在 [-1,1] 内（clamp 有效）。"""
        # 极端对方值不破有界
        for v_i, a_i in [(-1.0, -1.0), (1.0, 1.0), (-0.9, 0.9)]:
            for w_c in [0.1, 0.3, 1.0]:
                state = _state_with_stim(
                    goal_congruence=0.0,
                    interlocutor_affect=(v_i, a_i),
                    contagion_alpha=w_c,
                )
                out = AppraisalAgent()(state)
                v_out, a_out = out["prior_mu"]
                assert -1.0 <= v_out <= 1.0, f"v={v_out} 越界 (v_i={v_i}, w_c={w_c})"
                assert -1.0 <= a_out <= 1.0, f"a={a_out} 越界 (a_i={a_i}, w_c={w_c})"

    def test_contagion_both_dimensions_independent(self) -> None:
        """传染双维独立：valence 和 arousal 各自按差分公式更新、不互相影响。

        验证：传染后 (v_new, a_new) = clamp(v0+w*(v_i-v0), a0+w*(a_i-a0))
        """
        gc = 0.0
        v_i, a_i = 0.6, -0.5  # v正a负（避开 CARE 和替代喜悦触发区）
        w_c = 0.15
        state_no = _state_with_stim(goal_congruence=gc)
        mu_base = AppraisalAgent()(state_no)["prior_mu"]
        v0, a0 = mu_base

        state = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=w_c,
        )
        out = AppraisalAgent()(state)
        expected_v = clamp(v0 + w_c * (v_i - v0), -1.0, 1.0)
        expected_a = clamp(a0 + w_c * (a_i - a0), -1.0, 1.0)
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9)
        assert out["prior_mu"][1] == pytest.approx(expected_a, abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# 3. 开环（self 状态不影响 interlocutor_affect）
# ─────────────────────────────────────────────────────────────────


class TestOpenLoop:
    """开环验证：改 state 自身字段（mood/affect_sample 等）不改 interlocutor_affect 的注入值。

    interlocutor_affect 是外部注入的固定标量；AppraisalAgent 不回写它。
    """

    def test_change_mood_does_not_affect_interlocutor(self) -> None:
        """改变 state.mood → interlocutor_affect 仍然是注入的固定值（不被自身状态改变）。

        验证方式：两个 state 只有 mood 不同，interlocutor_affect 相同 → prior_mu 结果相同
        （因为 mood 不在 AppraisalAgent 的消费路径上，interlocutor_affect 固定）。
        """
        interlocutor = (-0.6, 0.5)
        state_no_mood = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.2,
            care_bias_alpha=0.3,
        )
        state_with_mood = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.2,
            care_bias_alpha=0.3,
            mood=(0.5, 0.3),  # 不同的自身 mood
        )
        out_no = AppraisalAgent()(state_no_mood)
        out_with = AppraisalAgent()(state_with_mood)
        # interlocutor_affect 固定相同 → prior_mu 应相同
        assert out_no["prior_mu"] == pytest.approx(out_with["prior_mu"], abs=1e-10), (
            "改变 mood 不应影响 interlocutor_affect 消费路径的结果"
        )

    def test_change_affect_sample_does_not_affect_interlocutor(self) -> None:
        """改变 state.affect_sample → 不影响 AppraisalAgent 输出（affect_sample 是下游字段）。"""
        interlocutor = (0.7, 0.4)  # 正面情绪
        state_a = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.1,
        )
        state_b = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.1,
            affect_sample=(-0.9, -0.9),  # 自身情感样本完全不同
        )
        out_a = AppraisalAgent()(state_a)
        out_b = AppraisalAgent()(state_b)
        assert out_a["prior_mu"] == pytest.approx(out_b["prior_mu"], abs=1e-10), (
            "改变 affect_sample 不影响 appraisal 先验（appraisal 早于采样）"
        )

    def test_interlocutor_affect_field_not_modified(self) -> None:
        """AppraisalAgent 不回写 interlocutor_affect 字段（开环保证）。"""
        interlocutor = (-0.5, 0.4)
        state = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.2,
        )
        out_dict = AppraisalAgent()(state)
        # 返回的增量 dict 不含 interlocutor_affect（节点不回写外部注入字段）
        assert "interlocutor_affect" not in out_dict, (
            "AppraisalAgent 不应在输出 dict 中包含/回写 interlocutor_affect"
        )
        # state 本身的 interlocutor_affect 保持原值（AppraisalAgent 无副作用）
        assert state.interlocutor_affect == interlocutor


# ─────────────────────────────────────────────────────────────────
# 4. CARE 触发（v_i < 0）
# ─────────────────────────────────────────────────────────────────


class TestCareTrigger:
    """CARE 触发：v_i<0 → CARE 偏置>0；v_i>0 → 不触发；v_i=0 边界不触发。

    方程：care_bias = care_bias_alpha * max(-v_i, 0.0) → 抬高 prior_mu valence。
    """

    def test_care_triggers_when_vi_negative(self) -> None:
        """v_i<0 + care_bias_alpha>0 → prior valence 被 relu(-v_i) 抬高。

        期望：v_new = clamp(v_after_contagion + care_alpha*(-v_i), -1, 1)
        """
        v_i, a_i = -0.6, 0.3  # 对方负面情绪
        care_alpha = 0.3
        w_c = 0.0  # 只测 CARE，关闭传染

        # baseline（无 interlocutor）
        state_no = _state_with_stim(goal_congruence=0.0)
        mu_base = AppraisalAgent()(state_no)["prior_mu"]
        v0 = mu_base[0]

        state = _state_with_stim(
            goal_congruence=0.0,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=w_c,
            care_bias_alpha=care_alpha,
            vicarious_alpha=0.0,
        )
        out = AppraisalAgent()(state)
        # 传染 Δ=0（w_c=0）→ v_after_contagion=v0；CARE_bias=care_alpha*max(-v_i,0)=care_alpha*0.6
        care_bias = care_alpha * max(-v_i, 0.0)
        expected_v = clamp(v0 + care_bias, -1.0, 1.0)
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
            f"CARE 触发后 valence 期望={expected_v:.4f}, 得={out['prior_mu'][0]:.4f}"
        )
        # 验证被抬高（care_bias > 0）
        assert out["prior_mu"][0] > mu_base[0] - 1e-9, (
            "CARE 触发应使 valence 上升（对方痛苦触发关怀）"
        )

    def test_care_does_not_trigger_when_vi_positive(self) -> None:
        """v_i>0 → CARE 不触发（relu(-v_i)=0 → care_bias=0）。

        对方正面情绪不触发 CARE；此时 interlocutor 偏置 = 0（关闭传染）。
        """
        gc = 0.3
        state_no = _state_with_stim(goal_congruence=gc)
        state_care = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.7, 0.3),  # v_i=0.7 > 0
            contagion_alpha=0.0,
            care_bias_alpha=0.4,
            vicarious_alpha=0.0,
            vicarious_threshold=0.9,  # 提高阈值防替代喜悦触发
        )
        out_no = AppraisalAgent()(state_no)
        out_care = AppraisalAgent()(state_care)
        # v_i>0 → CARE 不触发 → prior_mu 不变（传染也关）
        assert out_no["prior_mu"] == pytest.approx(out_care["prior_mu"], abs=1e-10), (
            "v_i>0 时 CARE 不触发，prior_mu 应与 baseline 相同"
        )

    def test_care_boundary_vi_zero_does_not_trigger(self) -> None:
        """v_i=0 边界：relu(-0)=0 → CARE 不触发（严格 < 0 触发）。"""
        gc = 0.2
        state_no = _state_with_stim(goal_congruence=gc)
        state_boundary = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.0, 0.5),  # v_i=0
            contagion_alpha=0.0,
            care_bias_alpha=0.4,
            vicarious_alpha=0.0,
        )
        out_no = AppraisalAgent()(state_no)
        out_b = AppraisalAgent()(state_boundary)
        # v_i=0 → max(-0, 0)=0 → care_bias=0 → prior_mu 不变
        assert out_no["prior_mu"] == pytest.approx(out_b["prior_mu"], abs=1e-10), (
            "v_i=0 边界不应触发 CARE"
        )

    def test_care_proportional_to_negative_vi(self) -> None:
        """CARE 偏置与 -v_i 成正比（v_i 越负、CARE 越高）。

        比较 v_i=-0.3 vs v_i=-0.7：后者 CARE 偏置更大 → valence 更高。
        """
        gc = 0.0
        care_alpha = 0.3
        state_mild = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(-0.3, 0.2),
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
        )
        state_strong = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(-0.7, 0.2),
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
        )
        out_mild = AppraisalAgent()(state_mild)
        out_strong = AppraisalAgent()(state_strong)
        assert out_strong["prior_mu"][0] >= out_mild["prior_mu"][0], (
            "v_i 越负 CARE 越强，valence 应更高"
        )


# ─────────────────────────────────────────────────────────────────
# 5. 替代喜悦触发
# ─────────────────────────────────────────────────────────────────


class TestVicariousJoy:
    """替代喜悦：v_i>threshold 且 a_i>0 → valence 上移；其他情况不触发。

    方程：vic_bias = vicarious_alpha * v_i（当 v_i>threshold 且 a_i>0）
    """

    def test_vicarious_joy_triggers_when_vi_above_threshold_and_ai_positive(self) -> None:
        """v_i>threshold 且 a_i>0 + vicarious_alpha>0 → prior_mu valence 上移。

        期望：v_new = clamp(v0 + vic_alpha*v_i, -1, 1)（无传染/CARE）
        """
        threshold = 0.3
        v_i, a_i = 0.7, 0.5  # v_i>threshold，a_i>0
        vic_alpha = 0.15
        gc = 0.0

        state_no = _state_with_stim(goal_congruence=gc)
        mu_base = AppraisalAgent()(state_no)["prior_mu"]
        v0 = mu_base[0]

        state = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=vic_alpha,
            vicarious_threshold=threshold,
        )
        out = AppraisalAgent()(state)
        vic_bias = vic_alpha * v_i
        expected_v = clamp(v0 + vic_bias, -1.0, 1.0)
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
            f"替代喜悦触发后 valence 期望={expected_v:.4f}, 得={out['prior_mu'][0]:.4f}"
        )
        # 验证 valence 确实上升
        assert out["prior_mu"][0] > mu_base[0] - 1e-9

    def test_vicarious_joy_does_not_trigger_vi_below_threshold(self) -> None:
        """v_i∈[0, threshold] → 替代喜悦不触发（不满足 v_i>threshold 条件）。"""
        threshold = 0.3
        gc = 0.1
        state_no = _state_with_stim(goal_congruence=gc)
        state_vic = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.2, 0.5),  # v_i=0.2 < threshold=0.3
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.2,
            vicarious_threshold=threshold,
        )
        out_no = AppraisalAgent()(state_no)
        out_vic = AppraisalAgent()(state_vic)
        assert out_no["prior_mu"] == pytest.approx(out_vic["prior_mu"], abs=1e-10), (
            "v_i<threshold 时替代喜悦不触发"
        )

    def test_vicarious_joy_does_not_trigger_when_ai_negative(self) -> None:
        """v_i>threshold 但 a_i<0 → 替代喜悦不触发（需要 a_i>0）。"""
        threshold = 0.3
        gc = 0.0
        state_no = _state_with_stim(goal_congruence=gc)
        state_vic = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.8, -0.3),  # v_i>threshold 但 a_i<0
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.2,
            vicarious_threshold=threshold,
        )
        out_no = AppraisalAgent()(state_no)
        out_vic = AppraisalAgent()(state_vic)
        assert out_no["prior_mu"] == pytest.approx(out_vic["prior_mu"], abs=1e-10), (
            "a_i<0 时替代喜悦不触发"
        )

    def test_vicarious_joy_does_not_trigger_when_ai_zero(self) -> None:
        """a_i=0 边界（代码用 > 0.0 判断）→ 不触发。"""
        threshold = 0.3
        gc = 0.0
        state_no = _state_with_stim(goal_congruence=gc)
        state_vic = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.8, 0.0),  # a_i=0，不满足 a_i>0
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.2,
            vicarious_threshold=threshold,
        )
        out_no = AppraisalAgent()(state_no)
        out_vic = AppraisalAgent()(state_vic)
        assert out_no["prior_mu"] == pytest.approx(out_vic["prior_mu"], abs=1e-10)

    def test_vicarious_joy_only_affects_valence_not_arousal(self) -> None:
        """替代喜悦：只抬高 valence，不影响 arousal（方程中无 arousal 项）。"""
        threshold = 0.3
        gc = 0.0
        state_no = _state_with_stim(goal_congruence=gc)
        mu_base = AppraisalAgent()(state_no)["prior_mu"]

        state_vic = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(0.7, 0.5),  # 触发替代
            contagion_alpha=0.0,
            care_bias_alpha=0.0,
            vicarious_alpha=0.15,
            vicarious_threshold=threshold,
        )
        out = AppraisalAgent()(state_vic)
        # arousal 不变（替代喜悦只改 valence）
        assert out["prior_mu"][1] == pytest.approx(mu_base[1], abs=1e-10), (
            "替代喜悦不应改变 arousal"
        )
        # valence 上升
        assert out["prior_mu"][0] > mu_base[0] - 1e-9


# ─────────────────────────────────────────────────────────────────
# 6. CARE / 替代喜悦互斥
# ─────────────────────────────────────────────────────────────────


class TestCareVicariousMutualExclusion:
    """CARE 与替代喜悦互斥：v<0 走 CARE、v>threshold 走替代；v∈[0,threshold] 两者均不触发。"""

    def test_negative_vi_triggers_only_care_not_vicarious(self) -> None:
        """v_i<0 → 走 CARE 路径，替代喜悦不触发（elif 互斥）。

        期望：
        - CARE_bias = care_alpha * max(-v_i, 0) > 0
        - vic_bias = 0（因 v_i<0，走了 elif 的 CARE 分支，替代喜悦 elif 不执行）
        """
        v_i, a_i = -0.5, 0.4  # v_i<0（触发 CARE）
        threshold = 0.3
        gc = 0.0
        care_alpha = 0.3
        vic_alpha = 0.2

        state_both = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
            vicarious_alpha=vic_alpha,
            vicarious_threshold=threshold,
        )
        state_care_only = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
            vicarious_alpha=0.0,  # 关闭替代
            vicarious_threshold=threshold,
        )
        out_both = AppraisalAgent()(state_both)
        out_care_only = AppraisalAgent()(state_care_only)
        # v_i<0 → 互斥 → vic_bias=0 → 两者结果相同
        assert out_both["prior_mu"] == pytest.approx(out_care_only["prior_mu"], abs=1e-10), (
            "v_i<0 应走 CARE、替代喜悦不触发（互斥），两者结果应相同"
        )

    def test_positive_vi_above_threshold_triggers_only_vicarious_not_care(self) -> None:
        """v_i>threshold → 走替代喜悦路径，CARE 不触发（v_i>0 → max(-v_i,0)=0）。"""
        v_i, a_i = 0.7, 0.5  # v_i>threshold=0.3，a_i>0
        threshold = 0.3
        gc = 0.0
        care_alpha = 0.3
        vic_alpha = 0.15

        state_both = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
            vicarious_alpha=vic_alpha,
            vicarious_threshold=threshold,
        )
        state_vic_only = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=(v_i, a_i),
            contagion_alpha=0.0,
            care_bias_alpha=0.0,  # 关闭 CARE（验证 CARE 本就不触发）
            vicarious_alpha=vic_alpha,
            vicarious_threshold=threshold,
        )
        out_both = AppraisalAgent()(state_both)
        out_vic_only = AppraisalAgent()(state_vic_only)
        # v_i>0 → CARE relu(-v_i)=0 → care_bias=0 → 两者结果相同
        assert out_both["prior_mu"] == pytest.approx(out_vic_only["prior_mu"], abs=1e-10), (
            "v_i>0 CARE 不触发（relu(-v_i)=0），两者结果应相同"
        )

    def test_vi_in_zero_to_threshold_triggers_neither(self) -> None:
        """v_i∈[0, threshold] → CARE 不触发（v_i>=0）且替代不触发（v_i<=threshold）。"""
        threshold = 0.3
        gc = 0.1
        state_no = _state_with_stim(goal_congruence=gc)
        # 在区间内的多个点
        for v_i in [0.0, 0.1, 0.2, threshold]:
            state = _state_with_stim(
                goal_congruence=gc,
                interlocutor_affect=(v_i, 0.5),
                contagion_alpha=0.0,
                care_bias_alpha=0.3,
                vicarious_alpha=0.2,
                vicarious_threshold=threshold,
            )
            out_no = AppraisalAgent()(state_no)
            out = AppraisalAgent()(state)
            assert out_no["prior_mu"] == pytest.approx(out["prior_mu"], abs=1e-10), (
                f"v_i={v_i}∈[0,{threshold}] 两者均不触发，prior_mu 应与 baseline 相同"
            )


# ─────────────────────────────────────────────────────────────────
# 7. 语义边界（WARN-5）
# ─────────────────────────────────────────────────────────────────


class TestSemanticBoundary:
    """语义边界（WARN-5）：interlocutor_affect 独立于 stim.goal_congruence。

    两者是不同来源、不同语义：
    - goal_congruence：自身 OCC 评价（「这事对我目标有多一致」），进 occ_prior 基线
    - interlocutor_affect：对方情绪 VAD（图外 appraise_text 独立估计），进共情偏置

    测试：构造 stim.goal_congruence 与 interlocutor_affect 取不同值 → 断言两者独立作用、互不混用。
    """

    def test_goal_congruence_changes_occ_prior_baseline_independently(self) -> None:
        """改变 goal_congruence → 改变 occ_prior 基线；interlocutor_affect 固定时偏置不变。

        验证：occ_prior 基线随 goal_congruence 变化，而 interlocutor 的偏置量只取决于
        interlocutor_affect 和 alpha 参数，与 goal_congruence 的值无关（两者独立叠加）。
        """
        interlocutor = (-0.5, 0.4)  # 固定对方情绪
        care_alpha = 0.3
        w_c = 0.0

        for gc in (-0.5, 0.0, 0.5):
            state = _state_with_stim(
                goal_congruence=gc,
                interlocutor_affect=interlocutor,
                contagion_alpha=w_c,
                care_bias_alpha=care_alpha,
            )
            out = AppraisalAgent()(state)
            # 手算期望（从 occ_prior 派生基线，然后叠加 CARE 偏置）
            mu_base, _, _ = occ_prior(gc, 0.0, 0.0, 0.5)
            v_i, _ = interlocutor
            care_bias = care_alpha * max(-v_i, 0.0)
            expected_v = clamp(mu_base[0] + care_bias, -1.0, 1.0)
            expected_a = clamp(mu_base[1], -1.0, 1.0)  # arousal 无偏置（care 只改 valence）
            assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
                f"gc={gc}: goal_congruence 与 interlocutor 独立叠加, valence 期望={expected_v:.4f}"
            )
            assert out["prior_mu"][1] == pytest.approx(expected_a, abs=1e-9), (
                f"gc={gc}: arousal 期望={expected_a:.4f}"
            )

    def test_interlocutor_affect_does_not_change_goal_congruence_semantics(self) -> None:
        """interlocutor_affect 变化不改变 stim.goal_congruence → occ_prior 基线路径独立。

        验证：对同一 goal_congruence，不同 interlocutor_affect 下，
        两个结果的差值恰好等于偏置差（纯偏置叠加，基线不变）。
        """
        gc = 0.4
        interlocutor_a = (-0.3, 0.2)
        interlocutor_b = (-0.7, 0.2)
        care_alpha = 0.3

        state_a = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=interlocutor_a,
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
        )
        state_b = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=interlocutor_b,
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
        )
        out_a = AppraisalAgent()(state_a)
        out_b = AppraisalAgent()(state_b)

        # 期望：偏置差 = care_alpha * (|v_b_neg| - |v_a_neg|) = 0.3*(0.7-0.3) = 0.12
        v_a_bias = care_alpha * max(-interlocutor_a[0], 0.0)  # 0.3*0.3 = 0.09
        v_b_bias = care_alpha * max(-interlocutor_b[0], 0.0)  # 0.3*0.7 = 0.21
        mu_base, _, _ = occ_prior(gc, 0.0, 0.0, 0.5)
        expected_v_a = clamp(mu_base[0] + v_a_bias, -1.0, 1.0)
        expected_v_b = clamp(mu_base[0] + v_b_bias, -1.0, 1.0)
        assert out_a["prior_mu"][0] == pytest.approx(expected_v_a, abs=1e-9)
        assert out_b["prior_mu"][0] == pytest.approx(expected_v_b, abs=1e-9)

    def test_interlocutor_affect_independent_of_occ_prior_basis(self) -> None:
        """验证 interlocutor_affect 与 goal_congruence 是独立作用的两个来源。

        构造：goal_congruence=0.8（正向目标一致性）+ interlocutor_affect=(-0.8, 0.5)（对方痛苦）
        → 自身基线因 gc=0.8 偏正 → CARE 偏置进一步抬升 valence
        两者叠加、互不污染（goal_congruence 决定基线，interlocutor 决定偏置）。
        """
        gc = 0.8
        interlocutor = (-0.8, 0.5)
        care_alpha = 0.3
        state = _state_with_stim(
            goal_congruence=gc,
            interlocutor_affect=interlocutor,
            contagion_alpha=0.0,
            care_bias_alpha=care_alpha,
        )
        out = AppraisalAgent()(state)
        # 手算：occ_prior 基线（gc=0.8 → 正向）+ CARE 偏置（v_i=-0.8 → bias=0.24）
        mu_base, _, _ = occ_prior(gc, 0.0, 0.0, 0.5)
        care_bias = care_alpha * max(-interlocutor[0], 0.0)
        expected_v = clamp(mu_base[0] + care_bias, -1.0, 1.0)
        assert out["prior_mu"][0] == pytest.approx(expected_v, abs=1e-9), (
            "goal_congruence 与 interlocutor_affect 独立叠加"
        )
        # 确认 interlocutor_affect 没有污染 occ_prior 内部参数（goal_congruence 还是 gc）
        state_no_interlocutor = _state_with_stim(goal_congruence=gc)
        out_no = AppraisalAgent()(state_no_interlocutor)
        # 差值 = CARE 偏置（独立叠加，非替换）
        diff = out["prior_mu"][0] - out_no["prior_mu"][0]
        assert diff == pytest.approx(
            care_bias if mu_base[0] + care_bias <= 1.0 else 1.0 - mu_base[0], abs=1e-9
        )


# ─────────────────────────────────────────────────────────────────
# 8. 热路径无 LLM/torch（静态检查）
# ─────────────────────────────────────────────────────────────────


class TestHotPathNoDependency:
    """静态验证 appraisal.py 共情块只用 math（clamp/max/+/*）、无 LLM/torch import。

    以 AST/inspect 方式检查模块级 import 和函数体，守住 affect 热路径红线。
    """

    def test_appraisal_module_no_torch_import(self) -> None:
        """appraisal.py 不 import torch（静态 AST 检查）。"""
        import src.agents.appraisal as appraisal_mod

        src_path = inspect.getfile(appraisal_mod)
        with open(src_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module.split(".")[0])
        assert "torch" not in imported_names, "appraisal.py 不应 import torch（热路径红线）"

    def test_appraisal_module_no_openai_import(self) -> None:
        """appraisal.py 不 import openai / LLM 相关模块（静态 AST 检查）。"""
        import src.agents.appraisal as appraisal_mod

        src_path = inspect.getfile(appraisal_mod)
        with open(src_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module.split(".")[0])
        forbidden = {"openai", "anthropic", "transformers", "langchain"}
        found = imported_names & forbidden
        assert not found, f"appraisal.py 不应 import LLM 库：{found}（热路径红线）"

    def test_appraisal_agent_callable_imports_are_pure_math(self) -> None:
        """AppraisalAgent.__call__ 源码只引用 affect_math 纯函数（clamp/occ_prior/cortisol_*）。

        验证共情偏置块（行 135-160）不引入新的 LLM/torch 调用。
        """
        import src.agents.appraisal as appraisal_mod

        src_path = inspect.getfile(appraisal_mod)
        with open(src_path, encoding="utf-8") as f:
            content = f.read()
        # 共情块相关关键词存在
        assert "interlocutor_affect" in content, "共情偏置块应存在"
        assert "contagion_alpha" in content, "传染系数应存在"
        assert "care_bias_alpha" in content, "CARE 系数应存在"
        assert "vicarious_alpha" in content, "替代喜悦系数应存在"
        # 共情块只用标准数学（不含 LLM/torch 调用关键字）
        empathy_forbidden_keywords = ["torch.", "openai.", "gpt", "lm.appraise", "language_model"]
        for kw in empathy_forbidden_keywords:
            assert kw not in content, f"appraisal.py 共情块不应包含 '{kw}'（热路径红线）"

    def test_appraisal_affect_math_imports_are_pure(self) -> None:
        """appraisal.py 从 affect_math 导入的都是纯数学函数（clamp/occ_prior/cortisol_*）。"""
        import src.agents.appraisal as appraisal_mod

        src_path = inspect.getfile(appraisal_mod)
        with open(src_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        # 找到 from src.agents.affect_math import ... 语句
        affect_math_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "affect_math" in node.module:
                    affect_math_imports.extend(alias.name for alias in node.names)
        assert affect_math_imports, "appraisal.py 应从 affect_math 导入纯函数"
        # 确认只有纯数学函数
        allowed = {
            "clamp",
            "occ_prior",
            "cortisol_step",
            "cortisol_trigger",
            "CORTISOL_CAP",
            "CORTISOL_IMPULSE",
            "CORTISOL_TAU_DECAY",
            "CORTISOL_THETA_GOAL",
            "CORTISOL_THETA_INTENSITY",
        }
        for name in affect_math_imports:
            assert name in allowed, f"appraisal.py 导入了非纯函数 '{name}'，不应引入 LLM/torch"

    def test_interlocutor_estimation_in_chat_driver_not_appraisal(self) -> None:
        """lexicon_appraise（interlocutor 估计侧）应在 chat_driver（编排层），不在 appraisal.py。

        验证：appraisal.py 不调用 lexicon_appraise（估计在图外、热路径只消费标量）。
        """
        import src.agents.appraisal as appraisal_mod

        src_path = inspect.getfile(appraisal_mod)
        with open(src_path, encoding="utf-8") as f:
            content = f.read()
        # appraisal.py 不含 lexicon_appraise 调用（估计在图外 chat_driver）
        assert "lexicon_appraise" not in content, (
            "appraisal.py 不应调用 lexicon_appraise（估计在图外，不在确定性节点内）"
        )


# ─────────────────────────────────────────────────────────────────
# 9. 贯通零回归（SessionConfig / env / build_chat_driver / to_state_flags）
# ─────────────────────────────────────────────────────────────────


class TestPassthroughZeroRegression:
    """贯通：SessionConfig 默认 4 字段；env → build_chat_driver → session.config。"""

    def test_session_config_default_contagion_alpha_zero(self) -> None:
        """SessionConfig() 默认 contagion_alpha=0.0（零回归）。"""
        cfg = SessionConfig()
        assert cfg.contagion_alpha == pytest.approx(0.0)

    def test_session_config_default_care_bias_alpha_zero(self) -> None:
        """SessionConfig() 默认 care_bias_alpha=0.0（零回归）。"""
        cfg = SessionConfig()
        assert cfg.care_bias_alpha == pytest.approx(0.0)

    def test_session_config_default_vicarious_alpha_zero(self) -> None:
        """SessionConfig() 默认 vicarious_alpha=0.0（零回归）。"""
        cfg = SessionConfig()
        assert cfg.vicarious_alpha == pytest.approx(0.0)

    def test_session_config_default_vicarious_threshold(self) -> None:
        """SessionConfig() 默认 vicarious_threshold=0.3。"""
        cfg = SessionConfig()
        assert cfg.vicarious_threshold == pytest.approx(0.3)

    def test_to_state_flags_contains_four_tom_fields(self) -> None:
        """to_state_flags() 展开后包含 4 个 ToM 旋钮字段。"""
        flags = SessionConfig().to_state_flags()
        assert "contagion_alpha" in flags, "to_state_flags 应包含 contagion_alpha"
        assert "care_bias_alpha" in flags, "to_state_flags 应包含 care_bias_alpha"
        assert "vicarious_alpha" in flags, "to_state_flags 应包含 vicarious_alpha"
        assert "vicarious_threshold" in flags, "to_state_flags 应包含 vicarious_threshold"

    def test_to_state_flags_default_values(self) -> None:
        """to_state_flags() 的 4 个 ToM 字段默认值对应零回归。"""
        flags = SessionConfig().to_state_flags()
        assert flags["contagion_alpha"] == pytest.approx(0.0)
        assert flags["care_bias_alpha"] == pytest.approx(0.0)
        assert flags["vicarious_alpha"] == pytest.approx(0.0)
        assert flags["vicarious_threshold"] == pytest.approx(0.3)

    def test_env_contagion_alpha_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CONTAGION_ALPHA=0.2 → session.config.contagion_alpha == 0.2。"""
        from src.orchestration.chat_driver import build_chat_driver

        monkeypatch.setenv("ZERO_CONTAGION_ALPHA", "0.2")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-ca")
        assert driver.session.config.contagion_alpha == pytest.approx(0.2)

    def test_env_care_bias_alpha_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CARE_BIAS_ALPHA=0.3 → session.config.care_bias_alpha == 0.3。"""
        from src.orchestration.chat_driver import build_chat_driver

        monkeypatch.setenv("ZERO_CARE_BIAS_ALPHA", "0.3")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-cba")
        assert driver.session.config.care_bias_alpha == pytest.approx(0.3)

    def test_env_vicarious_alpha_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_VICARIOUS_ALPHA=0.15 → session.config.vicarious_alpha == 0.15。"""
        from src.orchestration.chat_driver import build_chat_driver

        monkeypatch.setenv("ZERO_VICARIOUS_ALPHA", "0.15")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-va")
        assert driver.session.config.vicarious_alpha == pytest.approx(0.15)

    def test_env_vicarious_threshold_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_VICARIOUS_THRESHOLD=0.5 → session.config.vicarious_threshold == 0.5。"""
        from src.orchestration.chat_driver import build_chat_driver

        monkeypatch.setenv("ZERO_VICARIOUS_THRESHOLD", "0.5")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-vt")
        assert driver.session.config.vicarious_threshold == pytest.approx(0.5)

    def test_all_tom_defaults_no_env_zero_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不设任何 ToM env → to_state_flags() 取默认值（零回归保持）。"""
        monkeypatch.delenv("ZERO_CONTAGION_ALPHA", raising=False)
        monkeypatch.delenv("ZERO_CARE_BIAS_ALPHA", raising=False)
        monkeypatch.delenv("ZERO_VICARIOUS_ALPHA", raising=False)
        monkeypatch.delenv("ZERO_VICARIOUS_THRESHOLD", raising=False)
        flags = SessionConfig().to_state_flags()
        assert flags["contagion_alpha"] == pytest.approx(0.0)
        assert flags["care_bias_alpha"] == pytest.approx(0.0)
        assert flags["vicarious_alpha"] == pytest.approx(0.0)
        assert flags["vicarious_threshold"] == pytest.approx(0.3)
        # 旧字段零回归抽查
        assert flags["regulation_enabled"] is False
        assert flags["workspace_enabled"] is False
        assert flags["affect_readout"] == "sample"

    def test_conversation_session_legacy_params_tom(self) -> None:
        """旧展开参数传 ToM 字段 → session.config 字段正确。"""
        from src.orchestration.runner import ConversationSession

        session = ConversationSession(
            thread_id="t",
            contagion_alpha=0.1,
            care_bias_alpha=0.25,
            vicarious_alpha=0.12,
            vicarious_threshold=0.45,
        )
        assert session.config.contagion_alpha == pytest.approx(0.1)
        assert session.config.care_bias_alpha == pytest.approx(0.25)
        assert session.config.vicarious_alpha == pytest.approx(0.12)
        assert session.config.vicarious_threshold == pytest.approx(0.45)

    def test_conversation_session_config_takes_priority_over_legacy_tom(self) -> None:
        """传 config= 时 ToM 字段优先于旧展开参数。"""
        from src.orchestration.runner import ConversationSession

        cfg = SessionConfig(contagion_alpha=0.2, care_bias_alpha=0.3)
        session = ConversationSession(
            thread_id="t",
            config=cfg,
            contagion_alpha=0.0,  # 旧参数应被忽略
            care_bias_alpha=0.0,  # 旧参数应被忽略
        )
        assert session.config.contagion_alpha == pytest.approx(0.2)
        assert session.config.care_bias_alpha == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────
# 10. 稳定性上界校验（W3/W4·数学席证：alpha 硬上界 + L1≤0.6，防破 mood 双稳）
# ─────────────────────────────────────────────────────────────────


class TestEmpathyStabilityBounds:
    """W3/W4：SessionConfig 拒超界共情系数（数学席证防总偏置破 mood 双稳/attitude 收敛）。"""

    def test_contagion_over_cap_rejected(self) -> None:
        """contagion_alpha>0.3（数学席硬上界）→ ValidationError。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionConfig(contagion_alpha=0.5)

    def test_care_vicarious_over_cap_rejected(self) -> None:
        """care>0.4 / vicarious>0.2 → ValidationError。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionConfig(care_bias_alpha=0.5)
        with pytest.raises(ValidationError):
            SessionConfig(vicarious_alpha=0.3)

    def test_negative_alpha_rejected(self) -> None:
        """负系数（ge=0）→ ValidationError。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionConfig(contagion_alpha=-0.1)

    def test_l1_sum_over_0_6_rejected(self) -> None:
        """各系数在上界内但 L1 和>0.6（0.3+0.4+0.2=0.9）→ model_validator 拒。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SessionConfig(contagion_alpha=0.3, care_bias_alpha=0.4, vicarious_alpha=0.2)

    def test_within_bounds_ok(self) -> None:
        """各系数在上界内且 L1=0.55≤0.6 → 构造成功。"""
        cfg = SessionConfig(contagion_alpha=0.25, care_bias_alpha=0.2, vicarious_alpha=0.1)
        assert cfg.contagion_alpha == pytest.approx(0.25)

    def test_state_interlocutor_affect_default_none(self) -> None:
        """AffectState() 默认 interlocutor_affect=None（零回归门控关）。"""
        state = AffectState()
        assert state.interlocutor_affect is None

    def test_state_tom_alpha_defaults_zero(self) -> None:
        """AffectState() 默认 contagion=care=vicarious=0.0，threshold=0.3（零回归）。"""
        state = AffectState()
        assert state.contagion_alpha == pytest.approx(0.0)
        assert state.care_bias_alpha == pytest.approx(0.0)
        assert state.vicarious_alpha == pytest.approx(0.0)
        assert state.vicarious_threshold == pytest.approx(0.3)

    def test_chat_driver_tom_fields_default_zero(self) -> None:
        """ChatDriver 构造（直接传参）时 ToM 字段默认 0/0.3（零回归）。"""
        from src.orchestration.chat_driver import ChatDriver
        from src.orchestration.runner import ConversationSession
        from src.storage.conversation_log import ConversationLog

        session = ConversationSession(thread_id="t")
        driver = ChatDriver(
            thread="t",
            lm=None,
            log=ConversationLog(":memory:"),
            session=session,
            history=[],
            attitude=(0.0, 0.0),
            mode="test",
            noise_std=0.0,
        )
        assert driver.contagion_alpha == pytest.approx(0.0)
        assert driver.care_bias_alpha == pytest.approx(0.0)
        assert driver.vicarious_alpha == pytest.approx(0.0)
        assert driver.vicarious_threshold == pytest.approx(0.3)
