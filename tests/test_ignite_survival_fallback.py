"""ignite survival_fallback 门控测试（张力 4 裁决·神经席 M3）。

覆盖：
- 零回归：survival_fallback=False 时全弱刺激保留 salience 最高者（与旧行为逐字一致）。
- 零回归：有流过阈时 survival_fallback 不影响结果。
- 开启：survival_fallback=True + 全弱刺激 + 含 survival → 选 survival（非 salience 最高）。
- 开启：survival_fallback=True + 全弱刺激 + 无 survival → 退回 max by salience（不崩）。
- affect_core 集成：ignition_survival_fallback=False 时输出与旧路径一致。
- affect_core 集成：ignition_survival_fallback=True 且全弱刺激 → survival 兜底被选中。
"""

from __future__ import annotations

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import SALIENCE_THRESHOLD, ignite, stream_salience
from src.orchestration.state import AffectState, Stimulus

# ---------- 辅助 ----------

# 弱流：salience 明显低于 SALIENCE_THRESHOLD（0.18）
_WEAK_MU = (0.01, 0.01)
_WEAK_PREC = (0.2, 0.2)

# 稍强的弱流（salience 仍低于阈，但高于上面）
_MID_MU = (0.05, 0.05)
_MID_PREC = (0.3, 0.3)

# 确认全为亚阈
assert stream_salience(_WEAK_MU, _WEAK_PREC) < SALIENCE_THRESHOLD
assert stream_salience(_MID_MU, _MID_PREC) < SALIENCE_THRESHOLD
assert stream_salience(_MID_MU, _MID_PREC) > stream_salience(_WEAK_MU, _WEAK_PREC)

# 强流：salience 明显超过阈值
_STRONG_MU = (0.8, 0.8)
_STRONG_PREC = (1.5, 1.5)
assert stream_salience(_STRONG_MU, _STRONG_PREC) >= SALIENCE_THRESHOLD


# ---------- 纯函数单测 ----------


def test_zero_regression_all_weak_keeps_highest_salience() -> None:
    """survival_fallback=False（默认）：全弱刺激保留 salience 最高者，与旧行为逐字一致。"""
    streams = [
        ("survival", _WEAK_MU, _WEAK_PREC),
        ("appraisal", _MID_MU, _MID_PREC),  # salience 更高
    ]
    _, names_default = ignite(streams)
    _, names_false = ignite(streams, survival_fallback=False)
    # 两次调用结果一致（均为旧行为）
    assert names_default == names_false
    # 选的是 salience 最高者（appraisal），而非 survival
    assert names_false == ["appraisal"]


def test_zero_regression_fired_streams_unaffected() -> None:
    """有流过阈时，survival_fallback=True/False 结果完全一致。"""
    streams = [
        ("survival", _WEAK_MU, _WEAK_PREC),
        ("appraisal", _STRONG_MU, _STRONG_PREC),  # 过阈
    ]
    _, names_off = ignite(streams, survival_fallback=False)
    _, names_on = ignite(streams, survival_fallback=True)
    assert names_off == names_on
    assert "appraisal" in names_off


def test_survival_fallback_picks_survival_when_all_weak() -> None:
    """survival_fallback=True + 全弱刺激 + 含 survival 流 → 选 survival（即使非最高 salience）。"""
    streams = [
        ("survival", _WEAK_MU, _WEAK_PREC),  # salience 较低
        ("appraisal", _MID_MU, _MID_PREC),  # salience 较高，但不是 survival
    ]
    _, names = ignite(streams, survival_fallback=True)
    # 应选 survival，而非 salience 最高的 appraisal
    assert names == ["survival"], f"期望 survival 兜底，实际 {names}"


def test_survival_fallback_no_survival_stream_falls_back_to_max() -> None:
    """survival_fallback=True + 全弱刺激 + streams 无 survival 流 → 退回 max salience（不崩）。"""
    streams = [
        ("appraisal", _WEAK_MU, _WEAK_PREC),
        ("value", _MID_MU, _MID_PREC),  # salience 最高
    ]
    _, names = ignite(streams, survival_fallback=True)
    assert names == ["value"], f"无 survival 流时应退回 max salience，实际 {names}"


def test_survival_fallback_empty_streams_handled() -> None:
    """空 streams 边界：survival_fallback=True 时 max([], ...) 不被触发（streams 非空前提）。
    本测试确认非空单流场景正常运行。"""
    streams = [("survival", _WEAK_MU, _WEAK_PREC)]
    terms, names = ignite(streams, survival_fallback=True)
    assert names == ["survival"]
    assert len(terms) == 1


# ---------- affect_core 集成测试 ----------


def _ws_state(**kw: object) -> AffectState:
    """全弱刺激 workspace 状态：features 设为很低，确保 survival 流也亚阈。"""
    return AffectState(
        stimulus=Stimulus(name="t", goal_congruence=0.0, intensity=0.05),
        features=[0.0, 0.0, 0.0, 0.05],  # 低 → fast_survival_prior 出低 salience
        prior_mu=(0.01, 0.01),  # 评价流也很弱
        prior_sigma=(0.45, 0.45),  # sigma 大 → precision 低
        reward=0.0,
        rpe=0.0,
        precision=0.05,
        workspace_enabled=True,
        rng_seed=42,
        **kw,
    )


def test_affect_core_default_fallback_false_zero_regression() -> None:
    """ignition_survival_fallback=False（默认）：全弱刺激保留 salience 最高流（旧行为）。
    survival 流 salience 低，但 appraisal 流同样很弱，比谁高取决于 stream_salience 数值。
    关键断言：ignited_streams 非空（不空播）且与 False 显式传参结果一致。"""
    out_default = AffectCoreAgent()(_ws_state())
    out_false = AffectCoreAgent()(_ws_state(ignition_survival_fallback=False))
    assert out_default["ignited_streams"] == out_false["ignited_streams"]
    assert len(out_default["ignited_streams"]) >= 1  # 不空播


def test_affect_core_survival_fallback_true_selects_survival() -> None:
    """ignition_survival_fallback=True + 全弱刺激 → survival 流作兜底广播。

    fast_survival_prior([0,0,0,0.05]) 给出 survival salience，
    与评价/价值流都亚阈时，开启 fallback 应选中 survival。
    """
    out = AffectCoreAgent()(_ws_state(ignition_survival_fallback=True))
    ignited = out["ignited_streams"]
    assert "survival" in ignited, (
        f"ignition_survival_fallback=True + 全弱刺激时应兜底 survival，实际 {ignited}"
    )


def test_affect_core_state_field_default_is_false() -> None:
    """AffectState.ignition_survival_fallback 默认 False，确保零回归。"""
    state = AffectState()
    assert state.ignition_survival_fallback is False
