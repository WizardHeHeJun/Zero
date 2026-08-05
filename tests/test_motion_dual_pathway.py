"""动作双通路（① 情绪直驱 / ② 意志调控）：镜像 ExpressionAgent 既有语义的断言。

核心性质三条：默认零回归（两头逐字相同）、压制真的减幅、**压不到零**（防情绪塌陷）。
"""

from __future__ import annotations

import statistics

from src.agents.motion_synth import (
    MIN_VOLUNTARY_LEAK,
    PARAM_ANGLE_X,
    PhaseState,
    generate_dual,
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
