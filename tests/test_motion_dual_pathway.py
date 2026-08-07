"""动作双通路（① 情绪直驱 / ② 意志调控）：镜像 ExpressionAgent 既有语义的断言。

核心性质三条：默认零回归（两头逐字相同）、压制真的减幅、**压不到零**（防情绪塌陷）。
"""

from __future__ import annotations

import statistics

from src.agents.motion_synth import (
    MIN_VOLUNTARY_LEAK,
    PARAM_ANGLE_X,
    Modulation,
    PhaseState,
    generate_dual,
    modulation_from_affect,
)


def _spread(frames: list[dict[str, object]]) -> float:
    values = []
    for frame in frames:
        params = frame["params"]
        assert isinstance(params, dict)
        values.append(float(params[PARAM_ANGLE_X]))
    return statistics.pstdev(values)


def test_default_leak_makes_both_heads_identical() -> None:
    """未开调节 + leak=1.0 → 两头**逐字相同**（对齐 ExpressionAgent 的零回归语义）。"""
    heads, _ = generate_dual((0.2, 0.6), None, 3000.0, PhaseState(noise_seed=5))
    assert heads["voluntary"] == heads["spontaneous"]


def test_regulated_affect_changes_voluntary_only() -> None:
    """给了 regulated_affect → 随意头随之变，非随意头不受影响（情绪原样泄漏）。"""
    baseline, _ = generate_dual((0.0, 0.9), None, 3000.0, PhaseState(noise_seed=6))
    regulated, _ = generate_dual((0.0, 0.9), (0.0, 0.1), 3000.0, PhaseState(noise_seed=6))
    assert regulated["spontaneous"] == baseline["spontaneous"]
    assert regulated["voluntary"] != baseline["voluntary"]


def test_suppression_reduces_voluntary_amplitude() -> None:
    """压制使随意头动作变小——「压着点动」的直接体现。"""
    heads, _ = generate_dual(
        (0.0, 0.9), (0.0, 0.2), 4000.0, PhaseState(noise_seed=7), voluntary_leak=0.3
    )
    assert _spread(heads["voluntary"]) < _spread(heads["spontaneous"])


def test_voluntary_never_flattens_to_zero() -> None:
    """⚠ 意志压不平情绪：leak 给 0 也会被钳到 `MIN_VOLUNTARY_LEAK`。

    压平 = 情绪塌陷，是本项目既定裁定明确反对的（Rinn 1984：意志可部分压制但不归零）。
    撤掉钳制这条会红。
    """
    heads, _ = generate_dual(
        (0.0, 0.9), (0.0, 0.9), 4000.0, PhaseState(noise_seed=8), voluntary_leak=0.0
    )
    assert _spread(heads["voluntary"]) > 0.0
    ratio = _spread(heads["voluntary"]) / _spread(heads["spontaneous"])
    assert ratio >= MIN_VOLUNTARY_LEAK * 0.5  # 钳制真的生效，不是恰好非零


def test_both_heads_share_one_phase_advance() -> None:
    """两头共用同一相位推进——同一具身体，呼吸/眨眼节律不该分叉。"""
    heads, phase_out = generate_dual(
        (0.0, 0.5), (0.0, 0.2), 3000.0, PhaseState(noise_seed=9), voluntary_leak=0.5
    )
    assert phase_out.elapsed_ms == 3000.0
    assert len(heads["voluntary"]) == len(heads["spontaneous"])
    stamps_a = [f["t_ms"] for f in heads["spontaneous"]]
    stamps_b = [f["t_ms"] for f in heads["voluntary"]]
    assert stamps_a == stamps_b


def test_leak_above_one_is_clamped() -> None:
    """leak>1 不得放大情绪（意志只会压制，不会凭空增幅）。"""
    normal, _ = generate_dual((0.0, 0.6), None, 3000.0, PhaseState(noise_seed=10))
    over, _ = generate_dual(
        (0.0, 0.6), (0.0, 0.6), 3000.0, PhaseState(noise_seed=10), voluntary_leak=5.0
    )
    assert _spread(over["voluntary"]) == _spread(normal["voluntary"])


# -- modulation_voluntary（2026-08-07 真缺陷修复）--------------------------------
#
# 缺陷：directive 模式下把单个 Modulation 传给 generate_dual 时，两路共用同一份，
# voluntary 路丢掉了 regulated_affect 应有的调制系数，只剩 amplitude_scale*leak 的
# 整体缩放（速度/锐度不再随调节变化）。实证：affect=(-0.6,0.8) / regulated=(-0.2,0.1)
# / leak=0.6 / seed=42 / 8 秒，voluntary 峰值从 5.338 变成 9.375（高出 76%）。


def _peak(frames: list[dict[str, object]], key: str = PARAM_ANGLE_X) -> float:
    values = []
    for frame in frames:
        params = frame["params"]
        assert isinstance(params, dict)
        values.append(abs(float(params[key])))
    return max(values)


def test_modulation_voluntary_none_matches_no_regulated_affect_zero_regression() -> None:
    """modulation_voluntary 不传（None，默认）时逐字沿用现有行为——本参数引入前后一致。"""
    mod = modulation_from_affect(-0.6, 0.8)
    before, _ = generate_dual(
        (-0.6, 0.8),
        (-0.2, 0.1),
        4000.0,
        PhaseState(noise_seed=1),
        voluntary_leak=0.6,
        modulation=mod,
    )
    after, _ = generate_dual(
        (-0.6, 0.8),
        (-0.2, 0.1),
        4000.0,
        PhaseState(noise_seed=1),
        voluntary_leak=0.6,
        modulation=mod,
        modulation_voluntary=None,
    )
    assert before["spontaneous"] == after["spontaneous"]
    assert before["voluntary"] == after["voluntary"]


def test_modulation_voluntary_reproduces_regression_table() -> None:
    """钉住 code-reviewer 复现的那张表：directive 模式（两组独立系数）下 voluntary
    轨迹须与 synth 模式（modulation=None，两路各自现算）逐字相同——这条在修复前会红。
    """
    affect = (-0.6, 0.8)
    regulated = (-0.2, 0.1)
    leak = 0.6
    duration_ms = 8000.0

    # synth 模式现状：不传 modulation，两路各按自己的 (v,a) 内部自算。
    synth_heads, _ = generate_dual(
        affect, regulated, duration_ms, PhaseState(noise_seed=42), voluntary_leak=leak
    )

    # directive 模式（修复后）：两组系数分别由 affect / regulated_affect 算出，各喂各路。
    mod_spontaneous = modulation_from_affect(*affect)
    mod_voluntary = modulation_from_affect(*regulated)
    directive_heads, _ = generate_dual(
        affect,
        regulated,
        duration_ms,
        PhaseState(noise_seed=42),
        voluntary_leak=leak,
        modulation=mod_spontaneous,
        modulation_voluntary=mod_voluntary,
    )

    assert synth_heads["spontaneous"] == directive_heads["spontaneous"]
    assert synth_heads["voluntary"] == directive_heads["voluntary"]  # 修复前此断言会红
    # 双重确认：峰值也对得上（防"逐字相等"断言本身写错、恰好比出两个空列表这类假阳性）
    assert _peak(synth_heads["voluntary"]) == _peak(directive_heads["voluntary"])
    assert _peak(directive_heads["voluntary"]) < _peak(directive_heads["spontaneous"])


def test_modulation_voluntary_bug_reproduces_when_shared() -> None:
    """反向锁：若两路共用同一份 modulation（旧的错误用法），voluntary 确实会偏离
    synth 基线——证明上面那条修复测试测的是真问题，不是巧合过关。
    """
    affect = (-0.6, 0.8)
    regulated = (-0.2, 0.1)
    leak = 0.6
    duration_ms = 8000.0

    synth_heads, _ = generate_dual(
        affect, regulated, duration_ms, PhaseState(noise_seed=42), voluntary_leak=leak
    )
    shared_mod = modulation_from_affect(*affect)
    buggy_heads, _ = generate_dual(
        affect,
        regulated,
        duration_ms,
        PhaseState(noise_seed=42),
        voluntary_leak=leak,
        modulation=shared_mod,  # 旧错误用法：不传 modulation_voluntary，两路共用一份
    )
    assert synth_heads["voluntary"] != buggy_heads["voluntary"]


def test_short_circuit_still_applies_without_modulation_voluntary() -> None:
    """无调节（regulated_affect=affect）+ leak>=1.0 + modulation_voluntary=None 时，
    仍走"直接复用 spontaneous 对象"的短路分支——新增参数不破坏既有零回归优化。
    """
    heads, _ = generate_dual((0.1, 0.5), None, 3000.0, PhaseState(noise_seed=2))
    assert heads["voluntary"] is heads["spontaneous"]


def test_short_circuit_yields_to_explicit_modulation_voluntary() -> None:
    """⚠ 即便 source==affect 且 leak>=1.0，若调用方显式传了不同的 modulation_voluntary，
    也不得被短路分支静默吞掉——这是本参数区别于旧 modulation 参数的正确性要求。
    """
    base_mod = Modulation(amplitude=1.0, speed=1.0, onset_sharpness=0.5)
    distinct_mod = Modulation(amplitude=3.0, speed=3.0, onset_sharpness=1.0)
    heads, _ = generate_dual(
        (0.1, 0.5),
        None,  # regulated_affect=None → source==affect
        3000.0,
        PhaseState(noise_seed=2),
        modulation=base_mod,
        modulation_voluntary=distinct_mod,
    )
    assert heads["voluntary"] != heads["spontaneous"]
    assert _peak(heads["voluntary"]) > _peak(heads["spontaneous"])
