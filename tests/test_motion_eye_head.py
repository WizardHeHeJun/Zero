"""眼-头协同：眼先到位、头随后追赶、追上后眼球相对头部回正（前庭眼反射近似）。

2026-08-06 议会复核新增通道——此前眼球方向**完全静止**，被判为可信的不自然来源。
"""

from __future__ import annotations

from src.agents.motion_synth import (
    PARAM_EYE_L_X,
    PARAM_EYE_L_Y,
    PARAM_EYE_R_X,
    PARAM_EYE_R_Y,
    PhaseState,
    generate,
)


def _p(frame: dict[str, object]) -> dict[str, float]:
    params = frame["params"]
    assert isinstance(params, dict)
    return {k: float(v) for k, v in params.items()}


def test_eye_params_present_and_in_range() -> None:
    """四个眼球参数都在，且落在皮套实测值域 ±1 内（越界会被对面拒收）。"""
    frames, _ = generate(0.0, 0.3, 4000.0, PhaseState(noise_seed=4))
    for frame in frames:
        p = _p(frame)
        for key in (PARAM_EYE_L_X, PARAM_EYE_L_Y, PARAM_EYE_R_X, PARAM_EYE_R_Y):
            assert key in p
            assert -1.0 <= p[key] <= 1.0


def test_eyes_actually_move() -> None:
    """⚠ 眼球必须真的动——此前它恒为静止，正是被议会点名的不自然来源。

    把 EYE_AMPLITUDE_SCALE 置零会让这条红。
    """
    frames, _ = generate(0.0, 0.5, 6000.0, PhaseState(noise_seed=5))
    xs = [_p(f)[PARAM_EYE_L_X] for f in frames]
    assert max(xs) - min(xs) > 0.05, "眼球方向几乎不动"


def test_both_eyes_synchronised() -> None:
    """两眼同步（本模型不做斜视/辐辏）。"""
    frames, _ = generate(0.0, 0.4, 3000.0, PhaseState(noise_seed=6))
    for frame in frames:
        p = _p(frame)
        assert p[PARAM_EYE_L_X] == p[PARAM_EYE_R_X]
        assert p[PARAM_EYE_L_Y] == p[PARAM_EYE_R_Y]


def test_eye_offset_decays_when_head_settles() -> None:
    """核心机制：头部到位驻留后，眼球相对头部的偏移应回正到接近 0。

    这是「眼先到、头追上、眼回正」那条链的可验证末端。若实现成"眼睛跟着头一起慢慢动"，
    偏移量会恒定不为零，本条会红。
    """
    frames, _ = generate(0.0, -0.5, 8000.0, PhaseState(noise_seed=7))
    offsets = [abs(_p(f)[PARAM_EYE_L_X]) for f in frames]
    # 驻留期占多数 ⇒ 绝大多数帧的偏移应接近 0
    near_zero = sum(1 for v in offsets if v < 0.02)
    assert near_zero > len(offsets) * 0.5, "眼球偏移未回正——头到位后仍偏着"


def test_eye_leads_head_at_transition() -> None:
    """转移开始的瞬间，眼球先动而头部几乎未动 ⇒ 偏移出现峰值。

    验证「眼先到位」而不是两者同步移动。
    """
    frames, _ = generate(0.0, 0.6, 8000.0, PhaseState(noise_seed=8))
    peak_offset = max(abs(_p(f)[PARAM_EYE_L_X]) for f in frames)
    assert peak_offset > 0.1, "转移期眼球未领先于头部"


def test_deterministic() -> None:
    a, _ = generate(0.1, 0.2, 3000.0, PhaseState(noise_seed=9))
    b, _ = generate(0.1, 0.2, 3000.0, PhaseState(noise_seed=9))
    assert a == b
