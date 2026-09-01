"""口型 v2（音素级同步）单测——`PRP/lipsync-v2/tasks.md` T2 验收（G3 映射测试）。

覆盖：四呼典型字 + o/iong 反例 + b/p/m 闭唇；resolve 带内/带外两分支；τ 重排 Σ
不变（含退化情形）；SLEW 限幅；M4 floor 方向（挑 round() 会不同的边界值）；
M3 不变式（shape_open=0 ⇒ 乘积恒 0，即使 envelope 拉满）。
"""

from __future__ import annotations

import math

import pytest

from src.expression_out.lipsync_phoneme import (
    _VOWEL_SHAPE_TABLE,
    BETA,
    BILABIAL_CONSONANTS,
    DURATION_TOLERANCE_FRAMES,
    DURATION_TOLERANCE_RATIO,
    SLEW_RATE,
    TAU_MS,
    VOICED_MIN_OPENING_V2,
    _mix_energy,
    _phone_boundaries_ms,
    _shape_for_phones,
    _slew_limit_shapes,
    _tau_affine_reorder,
    build_phoneme_keyframes,
    frame_ms_from_sample_rate,
    resolve_phoneme_durations,
)

# ── frame_ms_from_sample_rate ───────────────────────────────────────────────


def test_frame_ms_from_sample_rate_matches_bertvits2_config() -> None:
    """design.md：`11.61ms = hop_length(512)/sample_rate(44100)`（本地 config 实证）。"""
    assert frame_ms_from_sample_rate(44100) == pytest.approx(11.6099773, abs=1e-4)


def test_frame_ms_from_sample_rate_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        frame_ms_from_sample_rate(0)


# ── 四呼典型字 + o/iong 反例（查表键=主元音音段，M2） ────────────────────────


def test_four_categories_typical_vowels_shape_sanity() -> None:
    """开口/齐齿/合口/撮口四呼各挑一典型韵母，开度方向应符合直觉。"""
    assert _VOWEL_SHAPE_TABLE["a"][0] > 0.7  # 开口呼：大开度
    assert _VOWEL_SHAPE_TABLE["i"][0] < 0.3  # 齐齿呼：小开度
    assert _VOWEL_SHAPE_TABLE["u"][0] < 0.3  # 合口呼：小开度、高圆唇
    assert _VOWEL_SHAPE_TABLE["u"][1] > 0.7
    assert _VOWEL_SHAPE_TABLE["v"][1] > 0.7  # 撮口呼：高圆唇


def test_o_and_iong_are_rounded_not_misjudged_by_hu_category() -> None:
    """M2 反例：`o`（零声母无韵头）、`iong`（齐齿呼标签下）主元音都是圆唇——
    若按"有无 u/v 韵头"这类四呼标签分类会误判成不圆唇，查表键=主元音本身则不会。
    """
    assert _VOWEL_SHAPE_TABLE["o"][1] > 0.7
    assert _VOWEL_SHAPE_TABLE["iong"][1] > 0.7
    # 同一错误模式的订正：iu 主元音是圆唇的 "ou"，非发端的 "i"。
    assert _VOWEL_SHAPE_TABLE["iu"][1] > 0.7


def test_bilabial_consonants_force_closed_lips() -> None:
    """M2：b/p/m 强制 open=0，不查表、不受后接韵母影响。"""
    shapes = _shape_for_phones(["b", "a"])
    assert shapes[0] == (0.0, 0.0)
    assert BILABIAL_CONSONANTS == frozenset({"b", "p", "m"})


def test_non_bilabial_consonant_targets_following_vowel() -> None:
    """其余声母（如 zh）目标=后接韵母目标（M2），非零声母无表可查时才退化闭合。"""
    shapes = _shape_for_phones(["zh", "ong"])
    assert shapes[0] == _VOWEL_SHAPE_TABLE["ong"]


def test_trailing_consonant_with_no_following_vowel_degrades_to_closed() -> None:
    """design §三简化 #4：句尾无后续目标，退化闭合（非新增行为，v1 同款）。"""
    shapes = _shape_for_phones(["a", "n"])
    assert shapes[-1] == (0.0, 0.0)


def test_build_phoneme_keyframes_bilabial_stays_closed_even_at_loud_envelope() -> None:
    """端到端：b/p/m 段即使包络拉满，MouthOpen 仍为 0（M2 与 M3 共同钉死）。"""
    phones = ["m", "a"]
    durations_ms = [TAU_MS, TAU_MS]
    envelope = [1.0] * 20
    frames = build_phoneme_keyframes(phones, durations_ms, envelope, 10.0, None)
    assert frames[0]["params"]["MouthOpen"] == 0.0  # type: ignore[index]


# ── resolve_phoneme_durations：M8 带内/带外 ──────────────────────────────────


def test_resolve_within_tolerance_scales_proportionally() -> None:
    frame_ms = 11.61
    durations = [10, 20, 30]  # 帧数
    total_ms = sum(durations) * frame_ms
    resolved = resolve_phoneme_durations(["a", "b", "c"], durations, frame_ms, total_ms)
    assert resolved is not None
    assert sum(resolved) == pytest.approx(total_ms)
    assert resolved[1] / resolved[0] == pytest.approx(2.0)  # 比例不变


def test_resolve_outside_tolerance_returns_none() -> None:
    frame_ms = 11.61
    durations = [10, 20, 30]
    total_ms = sum(durations) * frame_ms
    tolerance_ms = max(DURATION_TOLERANCE_FRAMES * frame_ms, DURATION_TOLERANCE_RATIO * total_ms)
    wav_ms = total_ms + tolerance_ms * 5  # 明显带外
    assert resolve_phoneme_durations(["a", "b", "c"], durations, frame_ms, wav_ms) is None


@pytest.mark.parametrize(
    "phones,durations",
    [
        ([], []),
        (["a"], []),
        (["a", "b"], [10]),
        (["a"], [0]),
        (["a"], [-1]),
    ],
)
def test_resolve_structural_failures_return_none(phones: list[str], durations: list[int]) -> None:
    assert resolve_phoneme_durations(phones, durations, 11.61, 1000.0) is None


# ── τ 仿射重排：Σ 不变 + 退化情形 ────────────────────────────────────────────


def test_tau_reorder_preserves_total_duration() -> None:
    durations = [30.0, 200.0, 40.0, 150.0]  # 两短两长
    reordered = _tau_affine_reorder(durations)
    assert sum(reordered) == pytest.approx(sum(durations))
    assert reordered[0] == pytest.approx(TAU_MS)
    assert reordered[2] == pytest.approx(TAU_MS)
    assert reordered[1] < 200.0  # 柔性段被收缩吸收超额
    assert reordered[3] < 150.0


def test_tau_reorder_no_short_phones_is_identity() -> None:
    durations = [200.0, 150.0]
    assert _tau_affine_reorder(durations) == durations


def test_tau_reorder_degenerate_all_short_no_flex_falls_back_to_original() -> None:
    """Σ_short_forced 超总时长且无柔性段可收缩——确定性回退：原样返回，Σ 仍不变。"""
    durations = [10.0, 10.0, 10.0]  # 全短，强制 TAU_MS 后 Σ 远超原总时长
    reordered = _tau_affine_reorder(durations)
    assert reordered == durations
    assert sum(reordered) == pytest.approx(sum(durations))


# ── 音素边界：math.floor 向早偏置（M4） ──────────────────────────────────────


def test_phone_boundaries_use_floor_not_round_or_ceil() -> None:
    """挑 round()（银行家舍入）与 floor() 结果不同的边界值：79.5 → round=80，floor=79。"""
    durations_ms = [79.5, 50.0]
    boundaries = _phone_boundaries_ms(durations_ms)
    assert boundaries[0] == 0
    assert boundaries[1] == 79
    assert boundaries[1] != round(79.5)
    assert boundaries[1] != math.ceil(79.5)


# ── SLEW_RATE 限幅（M5，非均匀关键帧） ───────────────────────────────────────


def test_slew_limits_shape_transition_rate() -> None:
    """构造跳变（a→u 极端形状差）：相邻帧斜率不得超过 SLEW_RATE。"""
    shapes = [(0.0, 0.0), (1.0, 1.0)]
    boundaries = [0, 10]  # dt=10ms
    limited = _slew_limit_shapes(shapes, boundaries)
    assert limited[0] == (0.0, 0.0)  # 首帧 dt=0（静音基线起点），强制闭合
    expected_delta = SLEW_RATE * 10
    assert limited[1][0] == pytest.approx(expected_delta)
    assert limited[1][1] == pytest.approx(expected_delta)
    assert limited[1][0] < 1.0  # 确认真的被限幅，没有直接跳到目标


def test_slew_allows_full_target_when_dt_is_large_enough() -> None:
    shapes = [(0.0, 0.0), (0.5, 0.5)]
    boundaries = [0, 10_000]  # dt 足够大，SLEW 不再是瓶颈
    limited = _slew_limit_shapes(shapes, boundaries)
    assert limited[1] == pytest.approx((0.5, 0.5))


# ── M3 能量混合不变式：shape_open=0 ⇒ 乘积恒 0 ───────────────────────────────


def test_mix_energy_zero_shape_open_is_invariant_even_at_full_envelope() -> None:
    assert _mix_energy(0.0, 0.0) == 0.0
    assert _mix_energy(0.0, 1.0) == 0.0  # 即使包络拉满


def test_mix_energy_floor_applies_only_to_envelope_factor() -> None:
    """地板只施加在包络因子——`open=1` 时乘积下限 = BETA + (1-BETA)*floor，非直接=floor。"""
    result = _mix_energy(1.0, 0.0)
    assert result == pytest.approx(BETA + (1.0 - BETA) * VOICED_MIN_OPENING_V2)


# ── build_phoneme_keyframes：双键 vs 单键（M6/M7） ──────────────────────────


def test_single_key_when_mouth_form_param_is_none() -> None:
    frames = build_phoneme_keyframes(["a"], [100.0], [0.5] * 10, 10.0, None)
    assert all(set(f["params"]) == {"MouthOpen"} for f in frames)  # type: ignore[arg-type]


def test_dual_key_when_mouth_form_param_is_set() -> None:
    frames = build_phoneme_keyframes(["a"], [100.0], [0.5] * 10, 10.0, "MouthPucker")
    assert all(set(f["params"]) == {"MouthOpen", "MouthPucker"} for f in frames)  # type: ignore[arg-type]


def test_empty_input_gives_empty_track() -> None:
    assert build_phoneme_keyframes([], [], [0.5], 10.0, None) == []
