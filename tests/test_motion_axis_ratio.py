"""三轴幅度比：同域实测（StayStill 待机 104.6 分钟）而非手调值。

改前三轴等幅，导致低头幅度与转头一样大——真人待机时低头只有转头的三分之一，
这正是真机观感反馈「低头比较不自然」的来源。本组测试把实测比例钉住，防被改回等幅。
"""

from __future__ import annotations

import statistics

from src.agents.motion_synth import (
    AXIS_AMPLITUDE_RATIO,
    PARAM_ANGLE_X,
    PARAM_ANGLE_Y,
    PARAM_ANGLE_Z,
    PhaseState,
    generate,
    initial_blink_ms,
)


def _spread(frames: list[dict[str, object]], key: str) -> float:
    vals = []
    for frame in frames:
        params = frame["params"]
        assert isinstance(params, dict)
        vals.append(float(params[key]))
    return statistics.pstdev(vals)


def _long_run(seed: int = 31) -> list[dict[str, object]]:
    """跑够长才能让三轴各自的分布稳定（单段太短，随机目标不够多）。"""
    phase = PhaseState(noise_seed=seed, next_blink_ms=initial_blink_ms(seed))
    frames: list[dict[str, object]] = []
    for _ in range(8):
        seg, phase = generate(0.0, 0.0, 10000.0, phase)
        frames.extend(seg)
    return frames


def test_ratio_constant_matches_measurement() -> None:
    """常量本身即实测值——改动它须同步改注释里的出处，不能悄悄调。"""
    assert AXIS_AMPLITUDE_RATIO == (1.0, 0.33, 0.19)


def test_three_axes_are_not_equal_amplitude() -> None:
    """⚠ 三轴**不得**等幅——等幅是改前的缺陷，直接对应「低头不自然」的观感反馈。"""
    frames = _long_run()
    yaw = _spread(frames, PARAM_ANGLE_X)
    pitch = _spread(frames, PARAM_ANGLE_Y)
    roll = _spread(frames, PARAM_ANGLE_Z)
    assert yaw > pitch * 1.5, f"yaw({yaw:.2f}) 未显著大于 pitch({pitch:.2f})"
    assert pitch > roll, f"pitch({pitch:.2f}) 未大于 roll({roll:.2f})"


def test_axis_order_matches_human_idle() -> None:
    """次序须是 yaw > pitch > roll —— 真人待机以左右张望为主。

    ⚠ 这与从**说话数据**（RAVDESS）量到的次序相反（那里 pitch 最大，是说话时点头
    beat gesture 的产物）。次序搞反 = 用错了数据域。
    """
    frames = _long_run(seed=77)
    yaw = _spread(frames, PARAM_ANGLE_X)
    pitch = _spread(frames, PARAM_ANGLE_Y)
    roll = _spread(frames, PARAM_ANGLE_Z)
    assert yaw > pitch > roll, f"次序错：yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f}"


def test_measured_ratio_is_approximately_reproduced() -> None:
    """输出的三轴比例应接近实测比例（容差宽——合成器还叠了呼吸/漂移/微颤）。"""
    frames = _long_run(seed=5)
    yaw = _spread(frames, PARAM_ANGLE_X)
    pitch = _spread(frames, PARAM_ANGLE_Y)
    assert 0.2 < pitch / yaw < 0.55, f"pitch/yaw={pitch / yaw:.2f} 偏离实测 0.33 过多"
