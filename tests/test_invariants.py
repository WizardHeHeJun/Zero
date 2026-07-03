"""跨轮涌现行为的形式化不变量测试（框架改进 B9 / P2-A）。

科学家议会（CS 席）+ 工程团队（架构师 P2 + 审查门 I-1）三方一致指出：两时间尺度 /
双稳等跨轮涌现行为只有端到端 smoke、缺形式化不变量护栏。本文件补 attitude 稳态收敛
（反棘轮 homeostasis）与 mood 双稳滞后（历史依赖不可逆）两类确定性不变量，作为将来
调参（rate/reversion/mood 增益）的回归护栏。Supervisor 节流次数契约与 Fact checkpoint
roundtrip 由 B9 子代理补在各自专测。
"""

from __future__ import annotations

from src.agents.affect_math import (
    ATTITUDE_RATE,
    ATTITUDE_REVERSION,
    MOOD_INERTIA,
    MOOD_SELF_GAIN,
    MOOD_SELF_K,
    attitude_step,
    mood_step,
)


def test_attitude_converges_to_analytic_steady_state() -> None:
    """恒定 stimulus 下 attitude 收敛到解析稳态 a*=rate·s/(rate+reversion)。"""
    s = (0.8, 0.0)
    a = (0.0, 0.0)
    for _ in range(500):
        a = attitude_step(a, s)
    expected = ATTITUDE_RATE * s[0] / (ATTITUDE_RATE + ATTITUDE_REVERSION)
    assert abs(a[0] - expected) < 1e-3


def test_attitude_no_ratchet_beyond_stimulus() -> None:
    """反棘轮/homeostasis：持续同向 stimulus 不把 attitude 推过 |s|，且全程有界。"""
    s = (1.0, 0.0)
    a = (0.0, 0.0)
    for _ in range(1000):
        a = attitude_step(a, s)
        assert -1.0 <= a[0] <= 1.0
    assert a[0] < s[0]  # reversion 项封死单调漂移到极端


def test_mood_bistable_hysteresis_irreversible_under_weak_drive() -> None:
    """mood 陷入负盆后弱正 affect 拉不出（双稳滞后/历史依赖）。"""
    assert MOOD_SELF_GAIN * MOOD_SELF_K > 1.0 - MOOD_INERTIA  # pitchfork 条件
    m = (0.0, 0.0)
    for _ in range(80):  # 强负 affect 推入负盆
        m = mood_step(m, (-1.0, -1.0))
    assert m[0] < -0.3  # 已落负吸引盆
    for _ in range(20):  # 施弱正 affect
        m = mood_step(m, (0.1, 0.1))
    assert m[0] < 0.0  # 仍停负盆、未翻正 → 滞后


def test_mood_symmetric_positive_basin() -> None:
    """对称性：强正 affect 落正盆，弱负扰动拉不出。"""
    m = (0.0, 0.0)
    for _ in range(80):
        m = mood_step(m, (1.0, 1.0))
    assert m[0] > 0.3
    for _ in range(20):
        m = mood_step(m, (-0.1, -0.1))
    assert m[0] > 0.0
