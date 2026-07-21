"""coping_potential 独立标量流单测（第一批，议会 2026-07-13）。

覆盖范围：
  1. 零回归：motivational_system(-0.6,0.6,coping_potential=0.0)=="rage"
  2. enabled=False 时 AppraisalAgent 不产 coping_potential_state
  3. enabled=True 时 coping_potential_state == control_appraisal（approx）
  4. clamp（2.0→1.0）
  5. 词典 rage/fear 分离三段（>0.3/中间/< -0.3）
  6. SessionConfig 默认 False + to_state_flags 带该键
  7. 静态断言：appraisal.py 的 coping 更新代码不含 "goal_congruence"（防回归）
  8. Stimulus.model_dump 含 control_appraisal
"""

from __future__ import annotations

import inspect

import pytest

from src.agents.appraisal import AppraisalAgent
from src.agents.emotion_lexicon import motivational_system
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ─────────────────────────────────────────────────────────────────
# 1. 零回归：coping_potential=0.0 时 (-v,+a) 一律 rage
# ─────────────────────────────────────────────────────────────────


class TestMotivationalSystemZeroRegression:
    """coping_potential=0.0（默认）时行为与改前逐字一致。"""

    def test_negative_v_positive_a_default_is_rage(self) -> None:
        """(-v,+a) + coping_potential=0.0 → rage（零回归保证）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0)
        assert result == "rage"

    def test_negative_v_positive_a_no_coping_arg_is_rage(self) -> None:
        """不传 coping_potential → 默认 0.0 → rage（向后兼容）。"""
        result = motivational_system(-0.6, 0.6)
        assert result == "rage"

    def test_positive_v_seeking(self) -> None:
        """(+v,+a) → seeking（不受 coping_potential 影响）。"""
        assert motivational_system(0.6, 0.6, coping_potential=0.5) == "seeking"

    def test_positive_v_negative_a_care(self) -> None:
        """(+v,-a) → care（不受 coping_potential 影响）。"""
        assert motivational_system(0.6, -0.6, coping_potential=-0.5) == "care"

    def test_negative_v_negative_a_panic_grief(self) -> None:
        """(-v,-a) → panic_grief（不受 coping_potential 影响）。"""
        assert motivational_system(-0.6, -0.6, coping_potential=0.5) == "panic_grief"

    def test_neutral_radius_returns_neutral(self) -> None:
        """r < NEUTRAL_RADIUS → neutral（与 coping_potential 无关）。"""
        assert motivational_system(0.0, 0.0, coping_potential=0.9) == "neutral"


# ─────────────────────────────────────────────────────────────────
# 2 & 3. AppraisalAgent：enabled=False 不产字段；enabled=True 产字段
# ─────────────────────────────────────────────────────────────────


def _make_state(
    *,
    control_appraisal: float = 0.4,
    coping_potential_enabled: bool = False,
) -> AffectState:
    """构造最小 AffectState 供 AppraisalAgent 测试用。"""
    stim = Stimulus(
        name="test", goal_congruence=0.0, intensity=0.5, control_appraisal=control_appraisal
    )
    return AffectState(stimulus=stim, coping_potential_enabled=coping_potential_enabled)


class TestAppraisalAgentCopingPotential:
    """AppraisalAgent 对 coping_potential_state 的增量产出测试。"""

    def test_disabled_does_not_produce_coping_state(self) -> None:
        """enabled=False → 返回 dict 不含 coping_potential_state（零回归）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state(coping_potential_enabled=False)
        out = agent(state)
        assert "coping_potential_state" not in out

    def test_enabled_produces_coping_state(self) -> None:
        """enabled=True → 返回 dict 含 coping_potential_state。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state(control_appraisal=0.4, coping_potential_enabled=True)
        out = agent(state)
        assert "coping_potential_state" in out

    def test_enabled_value_equals_control_appraisal(self) -> None:
        """enabled=True 时 coping_potential_state ≈ control_appraisal（直接 clamp 传透）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        ctrl = 0.4
        state = _make_state(control_appraisal=ctrl, coping_potential_enabled=True)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(ctrl)

    def test_enabled_negative_value(self) -> None:
        """enabled=True，control_appraisal=-0.7 → coping_potential_state≈-0.7。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        ctrl = -0.7
        state = _make_state(control_appraisal=ctrl, coping_potential_enabled=True)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(ctrl)

    def test_no_stimulus_returns_empty(self) -> None:
        """无 stimulus → 返回空 dict（不崩）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = AffectState(stimulus=None, coping_potential_enabled=True)
        out = agent(state)
        assert out == {}


# ─────────────────────────────────────────────────────────────────
# 4. clamp：control_appraisal 超界时被钳制到 [-1, 1]
# ─────────────────────────────────────────────────────────────────


class TestCopingPotentialClamp:
    """control_appraisal 超界时 coping_potential_state 应被 clamp 到 [-1,1]。"""

    def test_clamp_above_one(self) -> None:
        """control_appraisal=2.0 → coping_potential_state=1.0。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state(control_appraisal=2.0, coping_potential_enabled=True)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(1.0)

    def test_clamp_below_minus_one(self) -> None:
        """control_appraisal=-2.0 → coping_potential_state=-1.0。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state(control_appraisal=-2.0, coping_potential_enabled=True)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(-1.0)

    def test_boundary_exactly_one(self) -> None:
        """control_appraisal=1.0 → coping_potential_state=1.0（边界无损）。"""
        agent = AppraisalAgent(now_fn=lambda: 1000.0)
        state = _make_state(control_appraisal=1.0, coping_potential_enabled=True)
        out = agent(state)
        assert out["coping_potential_state"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────
# 5. 词典层 rage/fear 三段分离
# ─────────────────────────────────────────────────────────────────


class TestMotivationalSystemCopingThreeSections:
    """(-v,+a) 象限三段：>0.3→rage / <-0.3→fear / 中间→rage（保守默认）。"""

    # 阈值 0.3/-0.3 为议会初值（Smith & Ellsworth 1985 control 维），测试用硬编码对应议会决定值。
    THRESHOLD_HIGH = 0.3
    THRESHOLD_LOW = -0.3

    def test_above_high_threshold_is_rage(self) -> None:
        """coping_potential=0.5 > 0.3 → rage（高控制/趋近端）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.5)
        assert result == "rage"

    def test_exactly_above_threshold_is_rage(self) -> None:
        """coping_potential=0.31 (刚过 0.3) → rage。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.31)
        assert result == "rage"

    def test_below_low_threshold_gate_closed_is_rage(self) -> None:
        """coping_potential=-0.5 < -0.3 + fear_domain_enabled=False(默认) → rage（WARN-3 门关）。

        改前此处断言 fear；WARN-3 fear 专属门落地后，默认门关→保守 rage 是正确的议会裁定行为。
        """
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5)
        assert result == "rage"

    def test_below_low_threshold_gate_open_is_fear(self) -> None:
        """coping_potential=-0.5 < -0.3 + fear_domain_enabled=True → fear（门开正路）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.5, fear_domain_enabled=True)
        assert result == "fear"

    def test_exactly_below_threshold_gate_closed_is_rage(self) -> None:
        """coping=-0.31(刚过 -0.3) + fear_domain_enabled=False(默认) → rage（WARN-3 门关）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.31)
        assert result == "rage"

    def test_exactly_below_threshold_gate_open_is_fear(self) -> None:
        """coping_potential=-0.31(刚过 -0.3) + fear_domain_enabled=True → fear（门开正路）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.31, fear_domain_enabled=True)
        assert result == "fear"

    def test_mid_section_positive_is_rage(self) -> None:
        """coping_potential=0.1（中间段，正侧）→ rage（保守默认）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.1)
        assert result == "rage"

    def test_mid_section_zero_is_rage(self) -> None:
        """coping_potential=0.0（中间段，零）→ rage（零回归）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.0)
        assert result == "rage"

    def test_mid_section_negative_is_rage(self) -> None:
        """coping_potential=-0.1（中间段，负侧）→ rage（保守默认）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.1)
        assert result == "rage"

    def test_exactly_at_high_threshold_is_rage(self) -> None:
        """coping_potential=0.3（等于阈值，非严格大于）→ rage（中间段）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=0.3)
        assert result == "rage"

    def test_exactly_at_low_threshold_is_rage(self) -> None:
        """coping_potential=-0.3（等于阈值，非严格小于）→ rage（中间段）。"""
        result = motivational_system(-0.6, 0.6, coping_potential=-0.3)
        assert result == "rage"

    def test_high_cp_with_weak_valence_arousal(self) -> None:
        """(-v,+a) 弱信号但有效（r > NEUTRAL_RADIUS）+ coping=0.5 → rage。"""
        result = motivational_system(-0.2, 0.3, coping_potential=0.5)
        assert result == "rage"


# ─────────────────────────────────────────────────────────────────
# 6. SessionConfig 默认值 + to_state_flags 透传
# ─────────────────────────────────────────────────────────────────


class TestSessionConfigCopingPotential:
    """SessionConfig 默认 coping_potential_enabled=False，to_state_flags 含该键。"""

    def test_default_is_false(self) -> None:
        """SessionConfig() 默认 coping_potential_enabled=False。"""
        cfg = SessionConfig()
        assert cfg.coping_potential_enabled is False

    def test_to_state_flags_contains_key(self) -> None:
        """to_state_flags() 返回 dict 含 'coping_potential_enabled' 键。"""
        cfg = SessionConfig()
        flags = cfg.to_state_flags()
        assert "coping_potential_enabled" in flags

    def test_to_state_flags_default_value_false(self) -> None:
        """to_state_flags() 里 coping_potential_enabled 默认 False。"""
        cfg = SessionConfig()
        flags = cfg.to_state_flags()
        assert flags["coping_potential_enabled"] is False

    def test_enabled_true_propagates(self) -> None:
        """SessionConfig(coping_potential_enabled=True) 正确传递。"""
        cfg = SessionConfig(coping_potential_enabled=True)
        assert cfg.coping_potential_enabled is True
        flags = cfg.to_state_flags()
        assert flags["coping_potential_enabled"] is True


# ─────────────────────────────────────────────────────────────────
# 7. 静态断言：appraisal.py 的 coping 更新代码不含 "goal_congruence"
# ─────────────────────────────────────────────────────────────────


class TestAppraisalSourceNotGoalCongruence:
    """T2 裁决防回归：coping 更新代码路径绝不读 goal_congruence。"""

    def test_coping_update_block_does_not_reference_goal_congruence(self) -> None:
        """AppraisalAgent.__call__ 的 coping 更新区块源码不含 'goal_congruence'。

        提取 __call__ 源码中 coping_potential 区块（从 'coping 更新' 到 return），
        断言不含 goal_congruence 引用——防止来源被误接到共线的 OCC VA 路径。
        """
        source = inspect.getsource(AppraisalAgent.__call__)
        # 定位 coping_potential 更新区块起点关键词
        idx = source.find("coping_potential_enabled")
        assert idx >= 0, "源码应含 coping_potential_enabled 检查"
        coping_block = source[idx:]
        assert "goal_congruence" not in coping_block, (
            "coping 更新区块不得引用 goal_congruence（T2 裁决：来源须正交于 OCC VA 路径）"
        )


# ─────────────────────────────────────────────────────────────────
# 8. Stimulus.model_dump 含 control_appraisal
# ─────────────────────────────────────────────────────────────────


class TestStimulusControlAppraisal:
    """Stimulus 含 control_appraisal 字段，序列化正确。"""

    def test_model_dump_contains_control_appraisal(self) -> None:
        """Stimulus.model_dump() 含 'control_appraisal' 键。"""
        stim = Stimulus(name="test")
        d = stim.model_dump()
        assert "control_appraisal" in d

    def test_default_control_appraisal_is_none(self) -> None:
        """Stimulus 默认 control_appraisal=None（B3 前置改，absent cue 精度趋零）。"""
        stim = Stimulus(name="test")
        assert stim.control_appraisal is None

    def test_explicit_control_appraisal_roundtrips(self) -> None:
        """显式传入 control_appraisal=0.7 后 model_dump 含正确值。"""
        stim = Stimulus(name="test", control_appraisal=0.7)
        d = stim.model_dump()
        assert d["control_appraisal"] == pytest.approx(0.7)

    def test_negative_control_appraisal(self) -> None:
        """control_appraisal=-0.5 可正常构造与序列化。"""
        stim = Stimulus(name="test", control_appraisal=-0.5)
        assert stim.control_appraisal == pytest.approx(-0.5)


# ─────────────────────────────────────────────────────────────────
# 9. 不入图谱静态断言：SupervisorAgent 记忆写入不含 coping_potential
# ─────────────────────────────────────────────────────────────────


class TestCopingPotentialNotInGraph:
    """coping_potential_state 是运行态（进 Checkpointer），绝不写入长期记忆图谱。"""

    def test_supervisor_source_does_not_reference_coping_potential(self) -> None:
        """SupervisorAgent（唯一记忆 flush 点）源码不引用 coping_potential。

        coping_potential_state 只进 Checkpointer、不入图谱（CS 红线、同 cortisol 先例）。
        Supervisor 是唯一 memory.write/write_episode 点，其源码不应出现该字段。
        """
        import src.orchestration.supervisor as supervisor_mod

        source = inspect.getsource(supervisor_mod)
        assert "coping_potential" not in source, (
            "SupervisorAgent 记忆写入路径不得引用 coping_potential（运行态不入图谱）"
        )


# ─────────────────────────────────────────────────────────────────
# 10. 图内接线（W1 修复）：coping_potential_state 经 _appraisal_summary
#     真正影响 motivational_system 标签（防「空悬通路」回归）
# ─────────────────────────────────────────────────────────────────


class TestCopingPotentialAppraisalSummaryWiring:
    """coping_potential_state 经 language._appraisal_summary 透传至 motivational_system。

    code-reviewer W1：批 1 曾漏接线——AppraisalAgent 写了 state 但无节点传给词典层，
    特性运行时空悬。修复后 _appraisal_summary 逐字复刻 distinguish_fear 透传 state 值。
    """

    @staticmethod
    def _summary(coping_potential_state: float, *, fear_domain_enabled: bool = False) -> str:
        from src.agents.language import _appraisal_summary

        stim = Stimulus(name="t", goal_congruence=-0.5, intensity=0.6)
        state = AffectState(
            stimulus=stim,
            appraisal={"valence": -0.6, "arousal": 0.6},  # (-v,+a) 象限
            coping_potential_state=coping_potential_state,
            coping_potential_enabled=True,
            fear_domain_enabled=fear_domain_enabled,
        )
        return _appraisal_summary(state)

    def test_low_coping_summary_gate_closed_shows_rage(self) -> None:
        """coping_potential=-0.5 + fear_domain_enabled=False(默认) → rage（WARN-3 门关保守默认）。

        改前此处断言 fear；WARN-3 fear 专属门落地后，默认门关→回退 rage 是正确的议会裁定行为。
        零回归口径：fear 本就该默认关，此处改预期为 rage 是「正确的默认行为变更」而非破坏。
        """
        assert "动机系统=rage" in self._summary(-0.5, fear_domain_enabled=False)

    def test_low_coping_summary_gate_open_shows_fear(self) -> None:
        """coping_potential=-0.5 + fear_domain_enabled=True → fear（门开·路径二接线生效）。

        对称用例：显式开门后，低 coping 正常产 fear（接线验证）。
        """
        assert "动机系统=fear" in self._summary(-0.5, fear_domain_enabled=True)

    def test_high_coping_summary_shows_rage(self) -> None:
        """control_appraisal 高（0.5）→ 摘要动机系统=rage。"""
        assert "动机系统=rage" in self._summary(0.5)

    def test_default_coping_summary_shows_rage(self) -> None:
        """coping_potential_state=0.0（默认/关）→ rage（零回归，与旧 (-v,+a) 一致）。"""
        assert "动机系统=rage" in self._summary(0.0)
