"""口型 v2（音素级同步）与 `speech.py` 的接线测试——`PRP/lipsync-v2/tasks.md` T4。

G1（门关零回归/零污染）+ G2（无契约头静默降级）+ G3（M6/M7/M8 键集与降级判据）+
M9（双预算，纯算法一致性）六项，全程 fake transport/`_synthesize`（不依赖真网络/
真 Bert-VITS2/真渲染端）。`lipsync_phoneme.py` 本身的映射表/τ/SLEW/M3 单测在
`tests/test_lipsync_phoneme.py`，本文件只测 `speech.py` 的**接线**与端到端键集/降级
行为，以及 M9 的一条纯算法一致性钉子。
"""

from __future__ import annotations

import io
import json
import logging
import math
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

from src.expression_out.lipsync import V1_MOUTH_PARAMS
from src.expression_out.lipsync_phoneme import (
    TAU_MS,
    build_phoneme_keyframes,
    frame_ms_from_sample_rate,
    resolve_phoneme_durations,
)
from src.expression_out.speech import SynthResult, TtsSpeechSink, _wav_duration_and_rate
from tests.test_expression_speech import FakeTransport, _spoken, _wav

FPS = 20.0


def _sink(
    transport: Any,
    tmp_path: Path,
    *,
    lipsync_v2: bool = False,
    mouth_form_param: str | None = None,
) -> TtsSpeechSink:
    return TtsSpeechSink(
        transport=transport,
        server_url="http://127.0.0.1:5000/voice",
        speaker="test",
        fps=FPS,
        wav_dir=tmp_path / "tts",
        lipsync_v2=lipsync_v2,
        mouth_form_param=mouth_form_param,
    )


def _mouth_track_keys(transport: FakeTransport) -> list[set[str]]:
    plays = [c for c in transport.calls if c[0] == "speech_play"]
    assert len(plays) == 1, "本文件每用例只投递一段"
    return [set(kf["params"]) for kf in plays[0][1]["mouth_track"]]


# ── ①G1：门关——完全不读 header（而非读了不用）────────────────────────────────


async def test_g1_gate_closed_never_reads_header_even_if_present(tmp_path: Path) -> None:
    """架构决策 #2：`lipsync_v2=False` 时 `_synthesize` 完全不碰响应头。

    直接探针 `_synthesize` 的返回值（`phones`/`durations` 必须是 `None`，不是
    "解析出来但不用"）+ 端到端轨迹键集仍恰为 `V1_MOUTH_PARAMS`（即使
    `mouth_form_param` 也配了、header 也真的带了，双重前提下键集都不该越界）。
    """
    httpx = pytest.importorskip("httpx")
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, lipsync_v2=False, mouth_form_param="MouthPucker")

    def handler(request: Any) -> Any:
        payload = json.dumps({"phones": ["a"], "durations": [10]})
        return httpx.Response(200, content=_wav(), headers={"X-Phoneme-Durations": payload})

    sink.http_transport = httpx.MockTransport(handler)

    probe = await sink._synthesize("探针")
    assert probe.phones is None
    assert probe.durations is None

    await sink.connect()
    await _spoken(sink, transport, "你好呀")
    for keys in _mouth_track_keys(transport):
        assert keys == set(V1_MOUTH_PARAMS)
    await sink.aclose()


# ── ②G2：头缺失/JSON 坏/长度不等 → 回退 v1 且留一条 [lipsync-v2] 日志 ─────────


@pytest.mark.parametrize(
    "case,header_value",
    [
        ("missing_header", None),
        ("malformed_json", "{not-json"),
        ("length_mismatch", json.dumps({"phones": ["a", "b"], "durations": [10]})),
    ],
)
async def test_g2_bad_header_falls_back_to_v1_with_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    case: str,
    header_value: str | None,
) -> None:
    httpx = pytest.importorskip("httpx")
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, lipsync_v2=True, mouth_form_param="MouthPucker")

    def handler(request: Any) -> Any:
        headers = {"X-Phoneme-Durations": header_value} if header_value is not None else {}
        return httpx.Response(200, content=_wav(), headers=headers)

    sink.http_transport = httpx.MockTransport(handler)
    await sink.connect()
    with caplog.at_level(logging.DEBUG, logger="src.expression_out.speech"):
        await _spoken(sink, transport, "你好呀")

    for keys in _mouth_track_keys(transport):
        assert keys == set(V1_MOUTH_PARAMS), f"{case}：应回退 v1 单键，未越界到 form 键"
    assert any("[lipsync-v2]" in r.message for r in caplog.records), (
        f"{case}：应留一条 [lipsync-v2] 前缀日志"
    )
    await sink.aclose()


# ── ③④ 门开：M6 单键退化 / 双键契约 ───────────────────────────────────────────


def _consistent_phones_durations(wav_bytes: bytes) -> tuple[list[str], list[int]]:
    """给一段 wav 造一份"结构合法且落在 M8 带内"的音素契约（单音素覆盖整句）。"""
    wav_duration_ms, sample_rate = _wav_duration_and_rate(wav_bytes)
    frame_ms = frame_ms_from_sample_rate(sample_rate)
    return ["a"], [round(wav_duration_ms / frame_ms)]


async def test_single_key_when_mouth_form_param_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门开 + `mouth_form_param=None`：每帧键集恰为 `{"MouthOpen"}`（M6 单键退化）。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, lipsync_v2=True, mouth_form_param=None)
    wav_bytes = _wav(seconds=0.5)
    phones, durations = _consistent_phones_durations(wav_bytes)

    async def fake_synth(self: TtsSpeechSink, text: str) -> SynthResult:
        return SynthResult(wav_bytes=wav_bytes, phones=phones, durations=durations)

    monkeypatch.setattr(TtsSpeechSink, "_synthesize", fake_synth)
    await sink.connect()
    await _spoken(sink, transport, "你好呀")
    for keys in _mouth_track_keys(transport):
        assert keys == {"MouthOpen"}
    await sink.aclose()


async def test_dual_key_when_mouth_form_param_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门开 + `mouth_form_param="MouthPucker"`（占位假名）：每帧同时含两键（键集统一契约）。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, lipsync_v2=True, mouth_form_param="MouthPucker")
    wav_bytes = _wav(seconds=0.5)
    phones, durations = _consistent_phones_durations(wav_bytes)

    async def fake_synth(self: TtsSpeechSink, text: str) -> SynthResult:
        return SynthResult(wav_bytes=wav_bytes, phones=phones, durations=durations)

    monkeypatch.setattr(TtsSpeechSink, "_synthesize", fake_synth)
    await sink.connect()
    await _spoken(sink, transport, "你好呀")
    for keys in _mouth_track_keys(transport):
        assert keys == {"MouthOpen", "MouthPucker"}
    await sink.aclose()


# ── ⑤ M8 带外：整段回退 v1 ────────────────────────────────────────────────────


async def test_m8_out_of_tolerance_falls_back_whole_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """durations 总时长与 wav 差 >1%（且 >2 帧）：M8 判 None，整段回退 v1 单键
    ——即使 `mouth_form_param` 已配置（有能力出双键时也不出，证明确实整段回退，
    非"漏加了 form 键"这种局部退化）；`[lipsync-v2]` 日志佐证确实经过 M8 判据分支
    （而非碰巧结果一致）。
    """
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, lipsync_v2=True, mouth_form_param="MouthPucker")
    wav_bytes = _wav(seconds=0.5)  # 500ms

    async def fake_synth(self: TtsSpeechSink, text: str) -> SynthResult:
        # 200 帧 × 11.61ms ≈ 2322ms，远超 500ms wav 的 M8 容差。
        return SynthResult(wav_bytes=wav_bytes, phones=["a"], durations=[200])

    monkeypatch.setattr(TtsSpeechSink, "_synthesize", fake_synth)
    await sink.connect()
    with caplog.at_level(logging.WARNING, logger="src.expression_out.speech"):
        await _spoken(sink, transport, "你好呀")
    for keys in _mouth_track_keys(transport):
        assert keys == {"MouthOpen"}, "M8 带外应整段回退 v1，不应出现 form 键"
    assert any("[lipsync-v2]" in r.message and "M8" in r.message for r in caplog.records), (
        "应留一条点名 M8 的 [lipsync-v2] warning，证明确实走了判据分支而非碰巧同形"
    )
    await sink.aclose()


# ── ⑥ M9 双预算：纯算法一致性（非真机时延） ──────────────────────────────────


def _silence_wav(n_samples: int, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(array("h", [0] * n_samples).tobytes())
    return buf.getvalue()


def test_m9_double_budget_boundary_alignment() -> None:
    """M9：局部音素边界误差 ≤20ms、全局累计 ≤80ms（design.md 既有跨仓口径）。

    构造已知边界的假 phones/durations（每段时长均 ≥ TAU_MS，绕开 τ 仿射重排的
    位移，边界与"帧数×frame_ms"的理论值可直接比较）+ 精确匹配总时长的 fake wav
    （M8 带内、scale≈1），比对 `build_phoneme_keyframes` 产出的 `t_ms` 与独立算出
    的理论边界（floor(累计和)）——纯算法一致性钉子，非真机时延测量。
    """
    sample_rate = 44100
    frame_ms = frame_ms_from_sample_rate(sample_rate)
    phones = ["b", "a", "zh", "ong", "n"]
    frame_counts = [12, 20, 15, 25, 10]  # × frame_ms 均 > TAU_MS(80ms)：不触发 τ 重排
    durations_ms_true = [d * frame_ms for d in frame_counts]
    assert all(d >= TAU_MS for d in durations_ms_true), "构造前提：本用例不测 τ 重排位移"

    total_ms = sum(durations_ms_true)
    n_samples = round(total_ms / 1000.0 * sample_rate)
    wav_bytes = _silence_wav(n_samples, sample_rate)
    wav_duration_ms, wav_rate = _wav_duration_and_rate(wav_bytes)
    assert wav_rate == sample_rate

    resolved_ms = resolve_phoneme_durations(phones, frame_counts, frame_ms, wav_duration_ms)
    assert resolved_ms is not None, "构造前提：应落在 M8 带内"

    expected_boundaries: list[int] = []
    cumulative = 0.0
    for d in resolved_ms:
        expected_boundaries.append(math.floor(cumulative))
        cumulative += d

    envelope = [0.5] * 100
    frames = build_phoneme_keyframes(phones, resolved_ms, envelope, 10.0, None)
    actual_boundaries = [int(f["t_ms"]) for f in frames]  # type: ignore[call-overload]

    local_errors = [abs(a - e) for a, e in zip(actual_boundaries, expected_boundaries, strict=True)]
    assert max(local_errors) <= 20, f"局部边界误差超预算（M9）：{local_errors}"
    global_error = abs(actual_boundaries[-1] - expected_boundaries[-1])
    assert global_error <= 80, "全局累计边界误差超预算（M9）"
