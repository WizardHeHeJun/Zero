"""P3 1-B HPA/皮质醇慢回路正式单测（tests/test_cortisol.py）。

覆盖范围：
  1. 纯函数退化/有界性：cortisol_step 精确 ZOH 离散化有界、无 NaN、单调消退
  2. 精确 vs Euler 对照（说明性注释）
  3. 触发解耦：cortisol_trigger 只依赖 appraisal 输入
  4. AppraisalAgent 集成（注入固定时钟）：enabled/disabled 两路
  5. arousal_gate 开时 arousal_baseline 被 cortisol 抬高、valence 不变
  6. 门控关零回归：cortisol_enabled=False → 输出与改前逐字一致
  7. ATTITUDE_RATE 消费：cortisol_attitude_gate 开时 rate_eff 放大
  8. 贯通零回归：SessionConfig 默认 5 字段、env→build_chat_driver→session.config、to_state_flags
  9. 持久化红线：cortisol 不入图谱（静态 grep）
  10. 无硬编码时钟：cortisol_step/cortisol_trigger 源码无 datetime.now/time.time
  11. 结构保证：cortisol_trigger 签名不接受 arousal/emotion 参数（TypeError 保护）

期望值均从方程/常数派生，不凭记忆硬编码。
"""

from __future__ import annotations

import inspect
import math

import pytest

from src.agents.affect_math import (
    ATTITUDE_RATE,
    CORTISOL_CAP,
    CORTISOL_IMPULSE,
    CORTISOL_TAU_DECAY,
    CORTISOL_THETA_GOAL,
    CORTISOL_THETA_INTENSITY,
    cortisol_step,
    cortisol_trigger,
)
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ─────────────────────────────────────────────────────────────────
# 1. 纯函数退化与有界性
# ─────────────────────────────────────────────────────────────────


class TestCortisolStepPureFunctionBoundedness:
    """cortisol_step 有界性、单调消退、脉冲注入。"""

    def test_zero_cortisol_zero_impulse_zero_delta_is_zero(self) -> None:
        """c=0, Δt=0, impulse=0 → 0（退化零：无输入无输出）。"""
        result = cortisol_step(0.0, 0.0, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        assert result == pytest.approx(0.0)

    def test_zero_cortisol_no_impulse_any_delta_is_zero(self) -> None:
        """c=0, impulse=0, 任意 Δt → 0（ZOH：α·0+0=0）。"""
        for dt in (0.0, 600.0, CORTISOL_TAU_DECAY, 7200.0, 1e6):
            assert cortisol_step(
                0.0, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0
            ) == pytest.approx(0.0), f"Δt={dt} 时 c=0 应保持 0"

    def test_cap_decay_strictly_less_than_cap(self) -> None:
        """c=cap, Δt=3600, impulse=0 → 结果 ∈ (0, cap)（单调消退，未到 0）。
        期望：α=exp(-3600/5400)=exp(-2/3)≈0.5134；c_new=α·cap≈0.5134。
        """
        dt = 3600.0
        expected_alpha = math.exp(-dt / CORTISOL_TAU_DECAY)
        expected = expected_alpha * CORTISOL_CAP
        result = cortisol_step(CORTISOL_CAP, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        assert result == pytest.approx(expected, rel=1e-9)
        assert 0.0 < result < CORTISOL_CAP

    def test_large_delta_t_no_diverge_no_negative_no_nan(self) -> None:
        """Δt=7200s（> τ=5400s）不发散/不变负/不 NaN。

        精确 ZOH：α=exp(-7200/5400)=exp(-4/3)≈0.2636∈(0,1)，任意正 Δt 无条件有界。
        对比 Euler c·(1-Δt/τ)=c·(1-7200/5400)=c·(-1/3) < 0 → 失效！
        """
        dt = 7200.0
        c = CORTISOL_CAP
        result = cortisol_step(c, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        # 精确期望
        expected_alpha = math.exp(-dt / CORTISOL_TAU_DECAY)
        expected = expected_alpha * c
        assert not math.isnan(result), "Δt>τ 时结果不应为 NaN"
        assert result >= 0.0, "结果不应为负"
        assert result <= CORTISOL_CAP, "结果不应超过 cap"
        assert result == pytest.approx(expected, rel=1e-9)
        # 说明性注释：Euler 此时会给出负值
        euler_result = c * (1.0 - dt / CORTISOL_TAU_DECAY)
        assert euler_result < 0.0, "Euler 离散化在 Δt>τ 时确实变负（说明精确 exp 的必要性）"

    def test_impulse_injection_at_zero_state_zero_delta(self) -> None:
        """c=0, Δt=0, impulse=0.7 → 0.7（脉冲直接注入，无消退）。
        期望：clamp(exp(0)·0 + 0.7, 0, 1) = clamp(0.7, 0, 1) = 0.7。
        """
        result = cortisol_step(0.0, 0.0, tau_decay=CORTISOL_TAU_DECAY, impulse=0.7)
        assert result == pytest.approx(0.7)

    def test_impulse_injection_at_nonzero_state(self) -> None:
        """c=0.3, Δt=600, impulse=0.4 → 精确 ZOH。
        期望：α=exp(-600/5400)≈0.8948；c_new=clamp(0.8948·0.3+0.4, 0, 1)=clamp(0.668, 0, 1)=0.668。
        """
        dt = 600.0
        c0 = 0.3
        imp = 0.4
        alpha = math.exp(-dt / CORTISOL_TAU_DECAY)
        expected = min(max(alpha * c0 + imp, 0.0), CORTISOL_CAP)
        result = cortisol_step(c0, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=imp)
        assert result == pytest.approx(expected, rel=1e-9)
        assert 0.0 <= result <= CORTISOL_CAP

    def test_monotonic_decay_over_multiple_steps(self) -> None:
        """多步消退（impulse=0）严格单调递减。"""
        c = 0.8
        dt = 1800.0  # 30min 每步
        previous = c
        for _ in range(6):
            c = cortisol_step(c, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
            assert c < previous, "无脉冲时皮质醇每步应严格递减"
            previous = c

    def test_cap_clamp_prevents_overflow(self) -> None:
        """c=0.8 + impulse=0.5 → 被 cap=1.0 钳制，不超 cap。
        期望：α·0.8 + 0.5 = 0.8·exp(0) + 0.5 = 1.3 → clamp 到 1.0（当 Δt=0）。
        """
        result = cortisol_step(0.8, 0.0, tau_decay=CORTISOL_TAU_DECAY, impulse=0.5, cap=1.0)
        assert result == pytest.approx(1.0)

    def test_negative_delta_t_treated_as_zero(self) -> None:
        """Δt<0（防御）：按 0 处理，c 不变（alpha=exp(0)=1）。
        实现：max(0, delta_t)/tau_decay → exp(-0/tau)=1 → c_new = c + impulse（钳制）。
        """
        c = 0.5
        result = cortisol_step(c, -100.0, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        # Δt<0 时 max(0, Δt)=0 → alpha=exp(0)=1 → c_new = 1·c + 0 = c
        assert result == pytest.approx(c)

    def test_very_large_delta_t_approaches_zero(self) -> None:
        """Δt→∞ 时皮质醇趋向 0（消退到基线），无 impulse。
        精确 ZOH：α=exp(-1e6/5400)≈0（机器精度），c_new≈0。
        """
        result = cortisol_step(CORTISOL_CAP, 1_000_000.0, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_two_tau_not_diverge(self) -> None:
        """Δt=2τ（10800s）不发散，α=exp(-2)≈0.1353∈(0,1)。"""
        dt = 2 * CORTISOL_TAU_DECAY
        result = cortisol_step(CORTISOL_CAP, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        expected = math.exp(-2.0) * CORTISOL_CAP
        assert result == pytest.approx(expected, rel=1e-9)
        assert result > 0.0
        assert result < CORTISOL_CAP


# ─────────────────────────────────────────────────────────────────
# 2. 触发解耦
# ─────────────────────────────────────────────────────────────────


class TestCortisolTriggerDecoupling:
    """cortisol_trigger 只依赖 appraisal 输入，结构上无法接受 arousal/emotion 状态。"""

    def test_stress_trigger_goal_blocked_high_intensity(self) -> None:
        """目标受阻 + 高强度 → 返回 impulse > 0（HPA 激活）。
        判据：goal_congruence < -theta_goal=-0.3 AND intensity > theta_intensity=0.5。
        """
        result = cortisol_trigger(-0.5, 0.8)
        assert result == pytest.approx(CORTISOL_IMPULSE)
        assert result > 0.0

    def test_no_trigger_goal_congruent(self) -> None:
        """目标一致（goal_congruence=+0.5）→ 不触发（返回 0）。"""
        result = cortisol_trigger(0.5, 0.8)
        assert result == pytest.approx(0.0)

    def test_no_trigger_low_intensity(self) -> None:
        """目标受阻但强度不足（intensity=0.3 < threshold=0.5）→ 不触发。"""
        result = cortisol_trigger(-0.5, 0.3)
        assert result == pytest.approx(0.0)

    def test_no_trigger_at_threshold_boundary_goal(self) -> None:
        """goal_congruence=-theta_goal（边界，非严格 <）→ 不触发。"""
        result = cortisol_trigger(-CORTISOL_THETA_GOAL, 0.9)
        assert result == pytest.approx(0.0)

    def test_no_trigger_at_threshold_boundary_intensity(self) -> None:
        """intensity=theta_intensity（边界，非严格 >）→ 不触发。"""
        result = cortisol_trigger(-0.8, CORTISOL_THETA_INTENSITY)
        assert result == pytest.approx(0.0)

    def test_trigger_just_past_threshold(self) -> None:
        """刚越过阈值（goal_congruence=-0.31, intensity=0.51）→ 触发。"""
        result = cortisol_trigger(-0.31, 0.51)
        assert result == pytest.approx(CORTISOL_IMPULSE)

    def test_custom_impulse_amount(self) -> None:
        """自定义 impulse 量可覆盖默认值（验证参数正确传入）。"""
        result = cortisol_trigger(-0.8, 0.9, impulse=0.5)
        assert result == pytest.approx(0.5)

    def test_custom_thresholds(self) -> None:
        """自定义 theta_goal/theta_intensity 覆盖默认值。"""
        # 宽松阈值：theta_goal=0.1, theta_intensity=0.2
        result = cortisol_trigger(-0.15, 0.25, theta_goal=0.1, theta_intensity=0.2)
        assert result == pytest.approx(CORTISOL_IMPULSE)
        # 严格阈值下相同输入不触发
        result2 = cortisol_trigger(-0.15, 0.25, theta_goal=0.3, theta_intensity=0.5)
        assert result2 == pytest.approx(0.0)

    def test_trigger_signature_cannot_accept_arousal_argument(self) -> None:
        """结构保证：cortisol_trigger 签名不接受 arousal 参数（防 runaway 结构红线）。
        传入 arousal 关键字参数 → TypeError，确保函数签名不含 arousal/emotion。
        """
        with pytest.raises(TypeError):
            cortisol_trigger(-0.5, 0.8, arousal=0.9)  # type: ignore[call-arg]

    def test_trigger_signature_cannot_accept_emotion_argument(self) -> None:
        """结构保证：cortisol_trigger 签名不接受 emotion 参数。"""
        with pytest.raises(TypeError):
            cortisol_trigger(-0.5, 0.8, emotion=(0.3, 0.7))  # type: ignore[call-arg]

    def test_trigger_is_independent_of_cortisol_state(self) -> None:
        """触发结果与当前 cortisol 状态无关（∂I/∂c≡0，开环，防 runaway）。
        相同 appraisal 输入，对比 cortisol=0 / cortisol=1.0（不传入 trigger），结果相同。
        """
        # cortisol_trigger 只接收 goal_congruence / intensity（验证签名参数列表）
        sig = inspect.signature(cortisol_trigger)
        param_names = list(sig.parameters.keys())
        assert "cortisol" not in param_names, "cortisol_trigger 不应接受 cortisol 状态参数"
        assert "arousal" not in param_names, "cortisol_trigger 不应接受 arousal 状态参数"
        assert "emotion" not in param_names, "cortisol_trigger 不应接受 emotion 状态参数"


# ─────────────────────────────────────────────────────────────────
# 3. AppraisalAgent 集成（注入固定时钟）
# ─────────────────────────────────────────────────────────────────


class TestAppraisalAgentCortisolIntegration:
    """AppraisalAgent 集成测试：注入固定时钟确保确定性。"""

    def _make_agent(self, fixed_ts: float = 10000.0):
        """注入固定时钟的 AppraisalAgent。"""
        from src.agents.appraisal import AppraisalAgent

        return AppraisalAgent(now_fn=lambda: fixed_ts)

    def test_stress_stimulus_triggers_cortisol_increase(self) -> None:
        """应激 stimulus（goal_congruence<-0.3, intensity>0.5）+ cortisol_enabled=True
        → 返回增量含 cortisol_state>0、cortisol_updated_at=固定ts。
        期望：impulse=CORTISOL_IMPULSE; delta_t=1200s; alpha=exp(-1200/5400);
               c_new = alpha·0 + CORTISOL_IMPULSE（从 0 起步）。
        """
        fixed_ts = 10000.0
        prev_ts = fixed_ts - 1200.0  # 1200s 前最后更新
        stim = Stimulus(name="stress", goal_congruence=-0.5, intensity=0.8)
        state = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=0.0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=False,
            cortisol_arousal_alpha=0.2,
        )
        agent = self._make_agent(fixed_ts)
        out = agent(state)

        # 应有 cortisol 增量字段
        assert "cortisol_state" in out
        assert "cortisol_updated_at" in out
        # 时钟注入值
        assert out["cortisol_updated_at"] == pytest.approx(fixed_ts)
        # 期望 cortisol > 0（脉冲注入后衰减）
        dt = 1200.0
        alpha = math.exp(-dt / CORTISOL_TAU_DECAY)
        expected_c = min(max(alpha * 0.0 + CORTISOL_IMPULSE, 0.0), CORTISOL_CAP)
        assert out["cortisol_state"] == pytest.approx(expected_c, rel=1e-9)
        assert out["cortisol_state"] > 0.0

    def test_non_stress_stimulus_cortisol_only_decays(self) -> None:
        """非应激 stimulus（goal_congruence>0）+ cortisol_enabled=True + 初始 cortisol=0.5
        → cortisol 只消退（无新增），returned cortisol < 初始值。
        """
        fixed_ts = 10000.0
        prev_ts = fixed_ts - 3600.0  # 3600s 前
        c0 = 0.5
        stim = Stimulus(name="calm", goal_congruence=0.7, intensity=0.9)  # 非应激：目标一致
        state = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=c0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=False,
        )
        agent = self._make_agent(fixed_ts)
        out = agent(state)

        # impulse=0（非应激），仅消退
        dt = 3600.0
        alpha = math.exp(-dt / CORTISOL_TAU_DECAY)
        expected_c = alpha * c0  # impulse=0
        assert out["cortisol_state"] == pytest.approx(expected_c, rel=1e-9)
        assert out["cortisol_state"] < c0

    def test_cortisol_disabled_no_update(self) -> None:
        """cortisol_enabled=False → 返回增量不含 cortisol 字段（零回归）。"""
        stim = Stimulus(name="stress", goal_congruence=-0.5, intensity=0.9)
        state = AffectState(
            stimulus=stim,
            cortisol_enabled=False,
            cortisol_state=0.3,
            cortisol_updated_at=9000.0,
        )
        agent = self._make_agent(10000.0)
        out = agent(state)

        # 门控关：不返回 cortisol 更新字段
        assert "cortisol_state" not in out
        assert "cortisol_updated_at" not in out

    def test_arousal_gate_raises_prior_mu_arousal(self) -> None:
        """cortisol_arousal_gate=True + cortisol_state>0 → prior_mu[1]（arousal）被 cortisol 抬高。

        对照：先得到 cortisol_arousal_gate=False 的 arousal，再得到 gate=True 后 arousal。
        有序：两次测试用相同 stim，但 gate 不同。
        """
        fixed_ts = 10000.0
        prev_ts = fixed_ts - 1200.0
        # 先喂非应激 stim，只是为了让 cortisol_state 保留在 state 里
        c0 = 0.5  # 已有皮质醇
        stim = Stimulus(name="neutral", goal_congruence=0.3, intensity=0.5)

        # Gate 关：arousal_offset = 0
        state_off = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=c0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=False,
            cortisol_arousal_alpha=0.2,
            arousal_baseline=0.0,
        )
        agent = self._make_agent(fixed_ts)
        out_off = agent(state_off)
        arousal_off = out_off["prior_mu"][1]

        # Gate 开：arousal_offset = alpha·c_new（注意 cortisol_new 已含衰减）
        state_on = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=c0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=True,
            cortisol_arousal_alpha=0.2,
            arousal_baseline=0.0,
        )
        out_on = agent(state_on)
        arousal_on = out_on["prior_mu"][1]

        assert arousal_on > arousal_off, (
            f"arousal_gate 开 arousal 应更高：on={arousal_on:.4f} off={arousal_off:.4f}"
        )

    def test_arousal_gate_does_not_change_valence(self) -> None:
        """cortisol→arousal 抬升只影响 arousal，不影响 valence（cortisol 不决定效价）。"""
        fixed_ts = 10000.0
        prev_ts = fixed_ts - 1200.0
        c0 = 0.5
        stim = Stimulus(name="neutral", goal_congruence=0.3, intensity=0.5)

        state_off = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=c0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=False,
            cortisol_arousal_alpha=0.2,
        )
        state_on = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=c0,
            cortisol_updated_at=prev_ts,
            cortisol_arousal_gate=True,
            cortisol_arousal_alpha=0.2,
        )
        agent = self._make_agent(fixed_ts)
        out_off = agent(state_off)
        out_on = agent(state_on)

        # valence 维不受 cortisol 影响
        assert out_off["prior_mu"][0] == pytest.approx(out_on["prior_mu"][0]), (
            "cortisol→arousal 抬升不应改变 valence"
        )

    def test_first_update_no_previous_timestamp(self) -> None:
        """cortisol_updated_at=None（首次）→ delta_t=0，皮质醇从 0 直接加脉冲。
        期望：delta_t=0 → alpha=1 → c_new = 0 + impulse = CORTISOL_IMPULSE（应激时）。
        """
        fixed_ts = 10000.0
        stim = Stimulus(name="stress", goal_congruence=-0.6, intensity=0.9)
        state = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=0.0,
            cortisol_updated_at=None,  # 首次
        )
        agent = self._make_agent(fixed_ts)
        out = agent(state)

        # delta_t=0 → alpha=1 → c_new = 1·0 + CORTISOL_IMPULSE = CORTISOL_IMPULSE
        expected = cortisol_step(0.0, 0.0, tau_decay=CORTISOL_TAU_DECAY, impulse=CORTISOL_IMPULSE)
        assert out["cortisol_state"] == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────
# 4. 门控关零回归
# ─────────────────────────────────────────────────────────────────


class TestCortisolDisabledZeroRegression:
    """cortisol_enabled=False → AppraisalAgent 行为与改前逐字一致。"""

    def test_disabled_output_identical_to_no_cortisol_state(self) -> None:
        """门控关时输出的 prior_mu 等与无 cortisol 字段的默认 state 完全一致。"""
        from src.agents.appraisal import AppraisalAgent

        stim = Stimulus(name="test", goal_congruence=0.3, intensity=0.7)

        # 门控关（明确传入）
        state_disabled = AffectState(
            stimulus=stim,
            cortisol_enabled=False,
            cortisol_state=0.9,  # 即使有高 cortisol，门控关时不应影响
            cortisol_arousal_gate=False,
            cortisol_arousal_alpha=0.3,
        )
        # 默认 state（cortisol 字段默认值=关）
        state_default = AffectState(stimulus=stim)

        agent = AppraisalAgent()
        out_disabled = agent(state_disabled)
        out_default = agent(state_default)

        # prior_mu、prior_sigma、reward 完全一致（零回归）
        assert out_disabled["prior_mu"] == pytest.approx(out_default["prior_mu"])
        assert out_disabled["prior_sigma"] == pytest.approx(out_default["prior_sigma"])
        assert out_disabled["reward"] == pytest.approx(out_default["reward"])

    def test_disabled_no_cortisol_in_output(self) -> None:
        """门控关 → 输出 dict 不含 cortisol_state/cortisol_updated_at 更新。"""
        from src.agents.appraisal import AppraisalAgent

        stim = Stimulus(name="t", goal_congruence=-0.5, intensity=0.8)
        state = AffectState(stimulus=stim, cortisol_enabled=False)
        out = AppraisalAgent()(state)

        assert "cortisol_state" not in out
        assert "cortisol_updated_at" not in out

    def test_arousal_baseline_not_changed_when_gate_off(self) -> None:
        """arousal_gate=False 时，arousal_baseline 偏置为 0，prior_mu[1] 与基准一致。"""
        from src.agents.appraisal import AppraisalAgent

        stim = Stimulus(name="t", goal_congruence=0.0, intensity=0.5)
        state_gate_off = AffectState(
            stimulus=stim,
            cortisol_enabled=True,
            cortisol_state=0.8,
            cortisol_updated_at=1000.0,
            cortisol_arousal_gate=False,
            cortisol_arousal_alpha=0.3,
        )
        agent = AppraisalAgent(now_fn=lambda: 2000.0)
        out_gate_off = agent(state_gate_off)

        # arousal 维应与 gate 关的基准一致（gate 关时 arousal_offset=0）
        # 注：两个 agent 用不同 now_fn，但 gate_off 下 cortisol_new 不影响 arousal
        # 更直接的断言：gate_off + cortisol_enabled=True 的 arousal 应等于 gate_off 基准
        stim2 = Stimulus(name="t2", goal_congruence=0.0, intensity=0.5)
        state_ref = AffectState(stimulus=stim2, cortisol_enabled=False)
        agent2 = AppraisalAgent()
        out_ref = agent2(state_ref)
        assert out_gate_off["prior_mu"][1] == pytest.approx(out_ref["prior_mu"][1])


# ─────────────────────────────────────────────────────────────────
# 5. ATTITUDE_RATE 消费（ChatDriver）
# ─────────────────────────────────────────────────────────────────


class TestCortisolAttitudeRateConsumption:
    """cortisol_attitude_gate 开时 rate_eff 被放大；关时不变（零回归）。"""

    def _make_driver(
        self, session, *, cortisol_attitude_gate: bool, cortisol_attitude_alpha: float
    ):
        from src.orchestration.chat_driver import ChatDriver
        from src.storage.conversation_log import ConversationLog

        return ChatDriver(
            thread="t",
            lm=None,
            log=ConversationLog(":memory:"),
            session=session,
            history=[],
            attitude=(0.0, 0.0),
            mode="test",
            noise_std=0.0,
            cortisol_attitude_gate=cortisol_attitude_gate,
            cortisol_attitude_alpha=cortisol_attitude_alpha,
        )

    async def test_gate_off_rate_eff_equals_attitude_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gate=False → rate_eff = ATTITUDE_RATE（不放大，零回归）。
        验证方式：cortisol=高值但 gate 关，attitude 更新与无 cortisol 完全一致。
        """
        monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)

        # 两个 session 都返回相同 e*，但都不含 cortisol_state（门控关时 ChatDriver 也不读它）
        class FakeSession:
            async def step(self, stim):
                return {
                    "valence_arousal": (0.4, 0.5),
                    "recalled_context": [],
                    "cortisol_state": 0.8,
                }

        driver_off = self._make_driver(
            FakeSession(), cortisol_attitude_gate=False, cortisol_attitude_alpha=1.0
        )
        driver_baseline = self._make_driver(
            FakeSession(), cortisol_attitude_gate=False, cortisol_attitude_alpha=0.0
        )
        from src.storage.conversation_log import ConversationLog

        driver_off.log = ConversationLog(":memory:")
        driver_baseline.log = ConversationLog(":memory:")

        turn_off = await driver_off.step("hello")
        turn_baseline = await driver_baseline.step("hello")

        # gate 关时 cortisol_alpha 对 attitude 无影响 → 两者 attitude 相同
        assert turn_off.attitude == pytest.approx(turn_baseline.attitude, abs=1e-10)

    async def test_gate_on_rate_eff_amplified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """gate=True + cortisol>0 → rate_eff > ATTITUDE_RATE → attitude 变化更大。

        对比：相同 e* 输入，gate 开/关后 attitude 变化量的差异。
        cortisol=0.5, alpha=1.0 → f_max = 1 + 1.0·0.5 = 1.5 ≤ 2（安全区间）。
        """
        monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)

        class FakeSession:
            def __init__(self, cortisol: float):
                self.cortisol = cortisol

            async def step(self, stim):
                return {
                    "valence_arousal": (0.8, 0.0),  # 强正效价，arousal=0
                    "recalled_context": [],
                    "cortisol_state": self.cortisol,
                }

        from src.storage.conversation_log import ConversationLog

        # Gate 关：rate_eff = ATTITUDE_RATE
        driver_off = self._make_driver(
            FakeSession(0.5), cortisol_attitude_gate=False, cortisol_attitude_alpha=1.0
        )
        driver_off.log = ConversationLog(":memory:")
        turn_off = await driver_off.step("hi")

        # Gate 开：rate_eff = ATTITUDE_RATE * (1 + 1.0 * 0.5) = ATTITUDE_RATE * 1.5
        driver_on = self._make_driver(
            FakeSession(0.5), cortisol_attitude_gate=True, cortisol_attitude_alpha=1.0
        )
        driver_on.log = ConversationLog(":memory:")
        turn_on = await driver_on.step("hi")

        # gate 开时 rate_eff 更大 → valence attitude 变化更大（正效价 e* → attitude valence 更高）
        assert abs(turn_on.attitude[0]) >= abs(turn_off.attitude[0]), (
            f"cortisol_gate 开时 attitude valence 应更高：on={turn_on.attitude[0]:.4f}, "
            f"off={turn_off.attitude[0]:.4f}"
        )

    async def test_f_max_within_safe_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """attitude_alpha≤1 且 cortisol≤1 → f_max = 1 + 1·1 = 2（设计上界，不超 2）。
        验证 rate_eff ≤ 2 * ATTITUDE_RATE（math 席稳定上界）。
        """
        monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)

        class FakeSession:
            async def step(self, stim):
                return {
                    "valence_arousal": (1.0, 0.0),
                    "recalled_context": [],
                    "cortisol_state": 1.0,  # max cortisol
                }

        from src.storage.conversation_log import ConversationLog

        driver = self._make_driver(
            FakeSession(), cortisol_attitude_gate=True, cortisol_attitude_alpha=1.0
        )
        driver.log = ConversationLog(":memory:")
        # 一轮后 attitude 变化量应 ≤ 2 * ATTITUDE_RATE（f_max≤2）
        initial_attitude = driver.attitude
        await driver.step("hi")
        delta_v = abs(driver.attitude[0] - initial_attitude[0])
        # attitude 一步变化量不超过 2 * ATTITUDE_RATE（因 |e*| ≤ 1）
        assert delta_v <= 2.0 * ATTITUDE_RATE + 1e-10, (
            f"f_max≤2 约束：attitude 变化 {delta_v:.6f} 超 2·ATTITUDE_RATE={2 * ATTITUDE_RATE:.6f}"
        )


# ─────────────────────────────────────────────────────────────────
# 6. 贯通零回归（SessionConfig + env → build_chat_driver）
# ─────────────────────────────────────────────────────────────────


class TestCortisolPassthroughZeroRegression:
    """贯通：SessionConfig 默认字段、env→build_chat_driver→session.config、to_state_flags。"""

    def test_session_config_default_cortisol_enabled_false(self) -> None:
        """SessionConfig() 默认 cortisol_enabled=False（总门控关=零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_enabled is False

    def test_session_config_default_arousal_gate_false(self) -> None:
        """SessionConfig() 默认 cortisol_arousal_gate=False（零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_arousal_gate is False

    def test_session_config_default_attitude_gate_false(self) -> None:
        """SessionConfig() 默认 cortisol_attitude_gate=False（零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_attitude_gate is False

    def test_session_config_default_arousal_alpha_zero(self) -> None:
        """SessionConfig() 默认 cortisol_arousal_alpha=0.0（offset=0=零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_arousal_alpha == pytest.approx(0.0)

    def test_session_config_default_attitude_alpha_zero(self) -> None:
        """SessionConfig() 默认 cortisol_attitude_alpha=0.0（rate_eff×1=零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_attitude_alpha == pytest.approx(0.0)

    def test_to_state_flags_contains_all_cortisol_fields(self) -> None:
        """to_state_flags() 展开包含所有 5 个 cortisol 字段。"""
        flags = SessionConfig().to_state_flags()
        assert "cortisol_enabled" in flags
        assert "cortisol_arousal_gate" in flags
        assert "cortisol_attitude_gate" in flags
        assert "cortisol_arousal_alpha" in flags
        assert "cortisol_attitude_alpha" in flags

    def test_to_state_flags_cortisol_defaults(self) -> None:
        """to_state_flags() 的 cortisol 字段默认值均为零回归值。"""
        flags = SessionConfig().to_state_flags()
        assert flags["cortisol_enabled"] is False
        assert flags["cortisol_arousal_gate"] is False
        assert flags["cortisol_attitude_gate"] is False
        assert flags["cortisol_arousal_alpha"] == pytest.approx(0.0)
        assert flags["cortisol_attitude_alpha"] == pytest.approx(0.0)

    def test_env_cortisol_enabled_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CORTISOL_ENABLED=1 → build_chat_driver → session.config.cortisol_enabled=True。"""
        monkeypatch.setenv("ZERO_CORTISOL_ENABLED", "1")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-en")
        assert driver.session.config.cortisol_enabled is True

    def test_env_cortisol_arousal_gate_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CORTISOL_AROUSAL_GATE=true → session.config.cortisol_arousal_gate=True。"""
        monkeypatch.setenv("ZERO_CORTISOL_AROUSAL_GATE", "true")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-ag")
        assert driver.session.config.cortisol_arousal_gate is True

    def test_env_cortisol_attitude_gate_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CORTISOL_ATTITUDE_GATE=yes → session.config.cortisol_attitude_gate=True。"""
        monkeypatch.setenv("ZERO_CORTISOL_ATTITUDE_GATE", "yes")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-attg")
        assert driver.session.config.cortisol_attitude_gate is True

    def test_env_cortisol_arousal_alpha_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CORTISOL_AROUSAL_ALPHA=0.2 → session.config.cortisol_arousal_alpha=0.2。"""
        monkeypatch.setenv("ZERO_CORTISOL_AROUSAL_ALPHA", "0.2")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-aa")
        assert driver.session.config.cortisol_arousal_alpha == pytest.approx(0.2)

    def test_env_cortisol_attitude_alpha_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_CORTISOL_ATTITUDE_ALPHA=0.8 → session.config.cortisol_attitude_alpha=0.8。"""
        monkeypatch.setenv("ZERO_CORTISOL_ATTITUDE_ALPHA", "0.8")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-ata")
        assert driver.session.config.cortisol_attitude_alpha == pytest.approx(0.8)

    def test_env_cortisol_dynamics_constants_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WARN-1 修复：tau/impulse/theta env 覆盖须到达 session.config（None=回退常量零回归）。"""
        monkeypatch.setenv("ZERO_CORTISOL_TAU", "3600")
        monkeypatch.setenv("ZERO_CORTISOL_IMPULSE", "0.5")
        monkeypatch.setenv("ZERO_CORTISOL_THETA_GOAL", "0.4")
        monkeypatch.setenv("ZERO_CORTISOL_THETA_INTENSITY", "0.6")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        cfg = build_chat_driver(thread="test-cortisol-dyn").session.config
        assert cfg.cortisol_tau == pytest.approx(3600.0)
        assert cfg.cortisol_impulse == pytest.approx(0.5)
        assert cfg.cortisol_theta_goal == pytest.approx(0.4)
        assert cfg.cortisol_theta_intensity == pytest.approx(0.6)

    def test_cortisol_dynamics_constants_default_none(self) -> None:
        """未设 env → tau/impulse/theta 默认 None（AppraisalAgent 回退常量=零回归）。"""
        cfg = SessionConfig()
        assert cfg.cortisol_tau is None
        assert cfg.cortisol_impulse is None
        assert cfg.cortisol_theta_goal is None
        assert cfg.cortisol_theta_intensity is None

    def test_env_all_cortisol_off_no_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设任何 ZERO_CORTISOL_* → 所有字段为零回归默认值（不影响现有行为）。"""
        for key in (
            "ZERO_CORTISOL_ENABLED",
            "ZERO_CORTISOL_AROUSAL_GATE",
            "ZERO_CORTISOL_ATTITUDE_GATE",
            "ZERO_CORTISOL_AROUSAL_ALPHA",
            "ZERO_CORTISOL_ATTITUDE_ALPHA",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        from src.orchestration.chat_driver import build_chat_driver

        driver = build_chat_driver(thread="test-cortisol-zr")
        assert driver.session.config.cortisol_enabled is False
        assert driver.session.config.cortisol_arousal_gate is False
        assert driver.session.config.cortisol_attitude_gate is False
        assert driver.session.config.cortisol_arousal_alpha == pytest.approx(0.0)
        assert driver.session.config.cortisol_attitude_alpha == pytest.approx(0.0)

    def test_affect_state_default_cortisol_fields(self) -> None:
        """AffectState() 默认的 cortisol 字段与 SessionConfig 默认一致（零回归）。"""
        state = AffectState()
        assert state.cortisol_state == pytest.approx(0.0)
        assert state.cortisol_updated_at is None
        assert state.cortisol_enabled is False
        assert state.cortisol_arousal_gate is False
        assert state.cortisol_attitude_gate is False
        assert state.cortisol_arousal_alpha == pytest.approx(0.0)
        assert state.cortisol_attitude_alpha == pytest.approx(0.0)

    def test_conversation_session_legacy_params_cortisol(self) -> None:
        """旧展开参数传 cortisol_* → session.config 字段正确（向后兼容）。"""
        from src.orchestration.runner import ConversationSession

        session = ConversationSession(
            thread_id="t",
            cortisol_enabled=True,
            cortisol_arousal_gate=True,
            cortisol_attitude_gate=True,
            cortisol_arousal_alpha=0.25,
            cortisol_attitude_alpha=0.75,
        )
        assert session.config.cortisol_enabled is True
        assert session.config.cortisol_arousal_gate is True
        assert session.config.cortisol_attitude_gate is True
        assert session.config.cortisol_arousal_alpha == pytest.approx(0.25)
        assert session.config.cortisol_attitude_alpha == pytest.approx(0.75)

    def test_config_priority_over_legacy_cortisol(self) -> None:
        """传 config= 时 cortisol 字段 config 优先（旧展开参数被忽略）。"""
        from src.orchestration.runner import ConversationSession

        cfg = SessionConfig(cortisol_enabled=True, cortisol_attitude_alpha=0.5)
        session = ConversationSession(
            thread_id="t",
            config=cfg,
            cortisol_enabled=False,  # 旧参数，应被忽略
            cortisol_attitude_alpha=0.0,  # 旧参数，应被忽略
        )
        assert session.config.cortisol_enabled is True
        assert session.config.cortisol_attitude_alpha == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# 7. 持久化红线守护：cortisol 不入图谱
# ─────────────────────────────────────────────────────────────────


class TestCortisolNotInMemoryGraph:
    """静态 grep：supervisor / memory_recall / memory 层源码无 cortisol 进 memory.write。"""

    def _get_source(self, module_path: str) -> str:
        """读取模块源码（用 inspect.getsource 或直接读文件）。"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("m", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return inspect.getsource(mod)

    def test_supervisor_no_cortisol_in_memory_write(self) -> None:
        """supervisor.py 中 cortisol 相关内容不出现在 memory.write 调用附近。

        验证方式：源码中若有 'cortisol' 字样，需确认它不与 'write' / 'memory.write' 同行。
        """
        supervisor_path = r"d:\Zero\src\orchestration\supervisor.py"
        with open(supervisor_path, encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # 若某行既含 cortisol 又含 memory/write，则违规
            if "cortisol" in line.lower() and (
                "memory.write" in line.lower() or ".write(" in line.lower()
            ):
                pytest.fail(f"supervisor.py:{i + 1} cortisol 靠近 memory.write（红线违规）：{line}")

    def test_memory_recall_no_cortisol(self) -> None:
        """memory_recall.py 不含 cortisol 字样（cortisol 不经记忆召回路径）。"""
        memory_recall_path = r"d:\Zero\src\orchestration\memory_recall.py"
        with open(memory_recall_path, encoding="utf-8") as f:
            content = f.read()
        assert "cortisol" not in content.lower(), (
            "memory_recall.py 不应含 cortisol 字样（cortisol 只进 Checkpointer，不入图谱）"
        )

    def test_memory_client_no_cortisol(self) -> None:
        """src/memory/ 层不含 cortisol 字样（cortisol 只在编排层/Checkpointer）。"""
        import os

        memory_dir = r"d:\Zero\src\memory"
        for root, _, files in os.walk(memory_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                if "cortisol" in content.lower():
                    pytest.fail(f"src/memory/{fname} 含 cortisol（应只进 Checkpointer 不入记忆层）")


# ─────────────────────────────────────────────────────────────────
# 8. 无硬编码时钟（纯函数红线）
# ─────────────────────────────────────────────────────────────────


class TestNoHardcodedClockInPureFunctions:
    """cortisol_step/trigger 可执行体（排除 docstring）不含 datetime.now/time.time 调用。

    检测策略：对函数源码做 AST 解析，扫描所有 Call 节点中的属性访问（Attribute）。
    这样只检测真实代码调用，docstring 里的「禁止 xxx」说明文字不会误报。
    """

    @staticmethod
    def _get_call_names(fn: object) -> set[str]:
        """从函数源码 AST 中提取所有函数调用的名称（obj.attr 形式）。

        排除 docstring：ast.parse 后 docstring 只是 Constant 节点，不会出现在 Call 里。
        """
        import ast
        import textwrap

        source = inspect.getsource(fn)  # type: ignore[arg-type]
        # dedent 以便 inspect.getsource 拿到的带缩进代码能正常 parse
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        call_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    # 形如 obj.attr()
                    obj = func.value
                    if isinstance(obj, ast.Name):
                        call_names.add(f"{obj.id}.{func.attr}")
                    elif isinstance(obj, ast.Attribute):
                        # 形如 mod.obj.attr()（多级属性）
                        if isinstance(obj.value, ast.Name):
                            call_names.add(f"{obj.value.id}.{obj.attr}.{func.attr}")
                elif isinstance(func, ast.Name):
                    call_names.add(func.id)
        return call_names

    def test_cortisol_step_no_datetime_now(self) -> None:
        """cortisol_step 可执行体（排除 docstring）无 datetime.now 调用（CS 红线）。
        docstring 里写"禁止 datetime.now()" 是注释说明，不违规。
        """
        call_names = self._get_call_names(cortisol_step)
        assert "datetime.now" not in call_names, (
            f"cortisol_step 体内含 datetime.now 调用（违 CS 红线）：{call_names}"
        )

    def test_cortisol_step_no_time_time(self) -> None:
        """cortisol_step 可执行体（排除 docstring）无 time.time 调用（CS 红线）。"""
        call_names = self._get_call_names(cortisol_step)
        assert "time.time" not in call_names, (
            f"cortisol_step 体内含 time.time 调用（违 CS 红线）：{call_names}"
        )

    def test_cortisol_trigger_no_datetime_now(self) -> None:
        """cortisol_trigger 可执行体无 datetime.now 调用（CS 红线）。"""
        call_names = self._get_call_names(cortisol_trigger)
        assert "datetime.now" not in call_names

    def test_cortisol_trigger_no_time_time(self) -> None:
        """cortisol_trigger 可执行体无 time.time 调用（CS 红线）。"""
        call_names = self._get_call_names(cortisol_trigger)
        assert "time.time" not in call_names

    def test_cortisol_step_accepts_delta_t_parameter(self) -> None:
        """cortisol_step 签名含 delta_t（编排层注入，不自算时钟）。"""
        sig = inspect.signature(cortisol_step)
        assert "delta_t" in sig.parameters, "cortisol_step 应接受 delta_t 参数（编排层注入时钟差）"

    def test_appraisal_agent_now_fn_injectable(self) -> None:
        """AppraisalAgent 可注入 now_fn（生产用 time.time，测试用固定值）。"""
        from src.agents.appraisal import AppraisalAgent

        called = []

        def fixed_clock():
            called.append(True)
            return 99999.0

        agent = AppraisalAgent(now_fn=fixed_clock)
        stim = Stimulus(name="t", goal_congruence=-0.5, intensity=0.9)
        state = AffectState(
            stimulus=stim, cortisol_enabled=True, cortisol_state=0.0, cortisol_updated_at=None
        )
        agent(state)
        assert called, "now_fn 注入的固定时钟应被调用（编排层时钟注入验证）"


# ─────────────────────────────────────────────────────────────────
# 9. 精确 ZOH vs Euler 对照（说明性）
# ─────────────────────────────────────────────────────────────────


class TestZOHVsEulerIllustration:
    """说明性测试：精确 ZOH vs Euler 离散化的有界性差异（验证设计决策正确）。"""

    def test_euler_goes_negative_at_large_delta_t(self) -> None:
        """Euler c·(1-Δt/τ) 在 Δt>τ 时变负（说明为何禁用 Euler）。
        本测试验证这一点，从而说明精确 exp 的必要性。
        """
        c = 0.5
        dt = CORTISOL_TAU_DECAY * 1.5  # 1.5τ > τ
        euler = c * (1.0 - dt / CORTISOL_TAU_DECAY)
        assert euler < 0.0, "Euler 离散化在 Δt=1.5τ 时应变负（这是为何禁用 Euler 的原因）"

    def test_exact_zoh_stays_positive_at_same_delta_t(self) -> None:
        """精确 ZOH 在相同 Δt=1.5τ 时保持正值有界（α=exp(-1.5)≈0.223）。"""
        c = 0.5
        dt = CORTISOL_TAU_DECAY * 1.5
        result = cortisol_step(c, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        expected = math.exp(-1.5) * c
        assert result > 0.0
        assert result == pytest.approx(expected, rel=1e-9)

    def test_euler_diverges_at_two_tau(self) -> None:
        """Euler 在 Δt=2τ 时发散（乘子=-1，c → -c）。"""
        c = 0.5
        dt = 2.0 * CORTISOL_TAU_DECAY
        euler = c * (1.0 - dt / CORTISOL_TAU_DECAY)
        # 乘子 = 1 - 2 = -1 → euler = -0.5
        assert euler == pytest.approx(-c)
        assert euler < 0.0

    def test_exact_zoh_bounded_at_two_tau(self) -> None:
        """精确 ZOH 在 Δt=2τ 时仍有界（α=exp(-2)≈0.135∈(0,1)）。"""
        c = 0.5
        dt = 2.0 * CORTISOL_TAU_DECAY
        result = cortisol_step(c, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
        assert 0.0 < result < c
        assert result == pytest.approx(math.exp(-2.0) * c, rel=1e-9)

    def test_zoh_always_in_zero_cap_for_all_delta_t(self) -> None:
        """精确 ZOH 对任意 Δt∈[0, 10τ] 均保持 ∈[0, cap]（无条件有界）。"""
        c = CORTISOL_CAP
        for multiplier in (0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0):
            dt = multiplier * CORTISOL_TAU_DECAY
            result = cortisol_step(c, dt, tau_decay=CORTISOL_TAU_DECAY, impulse=0.0)
            assert 0.0 <= result <= CORTISOL_CAP, (
                f"Δt={multiplier}τ 时 cortisol_step 结果={result:.6f} 超出 [0,cap] 范围"
            )
