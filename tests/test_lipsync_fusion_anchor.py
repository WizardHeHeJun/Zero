"""口型融合规则锚点（speech-output v1）。

钉死的事实（两个方向，缺一不可）：

1. **语音只写嘴**：`lipsync` 输出的关键帧参数键 ⊆ `MOUTH_PARAMS`。
2. **情绪不写嘴**：`motion_synth.generate_dual` 的情绪动作流参数键 ∩ `MOUTH_PARAMS` = ∅。

两条都直接断言**投递 payload 的参数键集合**（不是代码文本——判据不取在代理量上），
且各配变异用例证明锚点能红（绿灯必须先证明它能红）。
"""

from __future__ import annotations

import io
import math
import wave
from array import array
from typing import Any

import pytest

from src.expression_out.lipsync import (
    MOUTH_PARAMS,
    energy_envelope,
    envelope_to_mouth_track,
    mouth_track_from_wav,
)

FPS = 20.0


def _wav(segments: list[tuple[float, float]], rate: int = 44100) -> bytes:
    """合成测试 wav（PCM 16-bit mono）。segments = [(秒数, 振幅0~1)]，振幅 0=静音。"""
    samples = array("h")
    for seconds, amplitude in segments:
        n = int(seconds * rate)
        for i in range(n):
            value = amplitude * math.sin(2 * math.pi * 220.0 * i / rate)
            samples.append(int(value * 32000))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _assert_only_mouth_keys(track: list[dict[str, Any]]) -> None:
    """锚点①的断言本体：轨迹里出现嘴部集合之外的键即红。"""
    assert track, "空轨迹 ⇒ 本用例没在测东西"
    for frame in track:
        extra = set(frame["params"]) - set(MOUTH_PARAMS)
        assert not extra, f"语音流写了嘴部集合之外的参数：{extra}"


def _assert_no_mouth_keys(keyframes: list[dict[str, Any]]) -> None:
    """锚点②的断言本体：情绪动作流出现嘴部键即红。"""
    assert keyframes, "空轨迹 ⇒ 本用例没在测东西"
    for frame in keyframes:
        hit = set(frame["params"]) & set(MOUTH_PARAMS)
        assert not hit, f"情绪动作流写了嘴部参数：{hit}"


# ── 锚点①：语音只写嘴 ────────────────────────────────────────────────────────


def test_mouth_track_only_writes_mouth_params() -> None:
    track = mouth_track_from_wav(_wav([(0.2, 0.0), (0.5, 0.8), (0.2, 0.0)]), FPS)
    _assert_only_mouth_keys(track)


def test_anchor_one_goes_red_on_leaked_key() -> None:
    """变异：往轨迹塞一个头部角度键，锚点①必须当场红。"""
    track = mouth_track_from_wav(_wav([(0.3, 0.5)]), FPS)
    _assert_only_mouth_keys(track)  # 正确实现恒绿（先证不误伤）
    poisoned = [dict(f, params={**f["params"], "ParamAngleX": 5.0}) for f in track]
    with pytest.raises(AssertionError):
        _assert_only_mouth_keys(poisoned)


# ── 锚点②：情绪不写嘴 ────────────────────────────────────────────────────────


def test_motion_stream_never_writes_mouth_params() -> None:
    """对着 `VtsSink._push_segment` 的真实合成调用取 payload（同参调用，非自拼样例）。"""
    from src.agents.motion_synth import (
        PhaseState,
        generate_dual,
        initial_blink_ms,
        modulation_from_affect,
    )

    phase = PhaseState(noise_seed=20260814, next_blink_ms=initial_blink_ms(20260814))
    heads, _ = generate_dual(
        (0.5, 0.6),
        None,
        2000.0,
        phase,
        voluntary_leak=1.0,
        fps=FPS,
        modulation=modulation_from_affect(0.5, 0.6),
    )
    _assert_no_mouth_keys(heads["voluntary"])
    _assert_no_mouth_keys(heads["spontaneous"])


def test_anchor_two_goes_red_on_mouth_leak() -> None:
    """变异：往情绪动作帧塞嘴部键，锚点②必须当场红。"""
    keyframes = [{"t_ms": 0, "params": {"ParamAngleX": 1.0}}]
    _assert_no_mouth_keys(keyframes)
    poisoned = [{"t_ms": 0, "params": {"ParamAngleX": 1.0, MOUTH_PARAMS[0]: 0.5}}]
    with pytest.raises(AssertionError):
        _assert_no_mouth_keys(poisoned)


# ── lipsync 函数本身的行为 ───────────────────────────────────────────────────


def test_track_shape_matches_motion_synth_contract() -> None:
    """关键帧形状与 `motion_synth` 同构：t_ms 整数从 0 起、按 fps 递增。"""
    track = mouth_track_from_wav(_wav([(0.5, 0.6)]), FPS)
    assert track[0]["t_ms"] == 0
    steps = [track[i + 1]["t_ms"] - track[i]["t_ms"] for i in range(len(track) - 1)]  # type: ignore[operator]
    assert all(step == 50 for step in steps), "20fps 帧距应恒为 50ms"
    assert all(isinstance(f["t_ms"], int) for f in track)


def test_silence_keeps_mouth_closed_and_speech_opens_it() -> None:
    """静音段全零（闭嘴）、有声段有开口——方向层锚点。"""
    track = mouth_track_from_wav(_wav([(0.3, 0.0), (0.4, 0.8), (0.3, 0.0)]), FPS)
    values = [float(f["params"][MOUTH_PARAMS[0]]) for f in track]  # type: ignore[index]
    n = len(values)
    head, mid, tail = values[: n // 4], values[n // 3 : 2 * n // 3], values[-n // 8 :]
    assert max(head) == 0.0, "前置静音就该闭嘴"
    assert max(mid) > 0.5, "有声段开口应显著"
    assert max(tail) < 0.2, "尾部静音应闭回去（release 平滑允许小残量）"
    assert all(0.0 <= v <= 1.0 for v in values)


def test_track_is_deterministic() -> None:
    wav = _wav([(0.2, 0.0), (0.3, 0.7)])
    assert mouth_track_from_wav(wav, FPS) == mouth_track_from_wav(wav, FPS)


def test_quiet_speech_still_opens_mouth() -> None:
    """按整句分位归一：轻声句也张得开嘴（不做绝对幅度门）。"""
    quiet = mouth_track_from_wav(_wav([(0.5, 0.1)]), FPS)
    assert max(float(f["params"][MOUTH_PARAMS[0]]) for f in quiet) > 0.8  # type: ignore[index]


def test_unsupported_wav_raises_value_error() -> None:
    """8-bit / 多声道超界给 ValueError（sink 层捕获降级；纯函数不吞）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8-bit
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 800)
    with pytest.raises(ValueError, match="16-bit"):
        energy_envelope(buf.getvalue(), FPS)


def test_empty_envelope_gives_empty_track() -> None:
    assert envelope_to_mouth_track([], FPS) == []
