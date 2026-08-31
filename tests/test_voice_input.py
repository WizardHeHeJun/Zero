"""语音输入（push-to-talk，2026-08-31 计划③）：工厂门控/fail-fast + 转写拼接契约。

硬件层（麦克风采集）不进 CI；模型经注入替身测转写拼接。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.orchestration.voice_input import VoiceInput, build_voice_input


def test_gate_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_ASR_INPUT", raising=False)
    assert build_voice_input() is None


def test_gate_on_missing_model_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_ASR_INPUT", "true")
    monkeypatch.delenv("ZERO_ASR_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ZERO_ASR_MODEL"):
        build_voice_input()


def test_gate_on_missing_deps_fails_fast_with_extra_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 asr extra 时报错须带安装指引（sys.modules 置 None 模拟缺依赖，环境无关确定性）。"""
    monkeypatch.setenv("ZERO_ASR_INPUT", "true")
    monkeypatch.setenv("ZERO_ASR_MODEL", "small")
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="asr"):
        build_voice_input()


def _fake_model(texts: list[str]) -> Any:
    def transcribe(audio: Any, language: str, **kwargs: Any) -> tuple[list[Any], Any]:
        segments = [SimpleNamespace(text=t) for t in texts]
        return segments, SimpleNamespace(language=language)

    return SimpleNamespace(transcribe=transcribe)


def test_transcribe_joins_segments_and_strips() -> None:
    vi = VoiceInput(model=_fake_model([" 今天天气", "不错。 "]))
    assert vi.transcribe(object()) == "今天天气不错。"


def test_transcribe_empty_segments_returns_empty() -> None:
    vi = VoiceInput(model=_fake_model([]))
    assert vi.transcribe(object()) == ""


class _FakeSd:
    """sounddevice 替身：可配默认输入是否可用与设备表（真机默认=-1 场景 2026-08-31 实测）。"""

    def __init__(self, *, default_ok: bool, devices: list[dict]) -> None:
        self.default_ok = default_ok
        self.devices = devices

    def query_devices(self, index: int | None = None, kind: str | None = None):  # noqa: ANN201
        if kind == "input":
            if self.default_ok:
                return self.devices[0]
            raise RuntimeError("Error querying device -1")
        if index is None:
            return self.devices
        return self.devices[index]


def test_resolve_device_default_available_returns_none() -> None:
    from src.orchestration.voice_input import _resolve_input_device

    sd = _FakeSd(default_ok=True, devices=[{"name": "mic", "max_input_channels": 2}])
    assert _resolve_input_device(sd) is None


def test_resolve_device_default_missing_falls_back_to_first_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真机场景：有麦克风但 Windows 默认输入=-1 → 自动选第一个有输入通道的设备。"""
    from src.orchestration.voice_input import _resolve_input_device

    monkeypatch.delenv("ZERO_ASR_INPUT_DEVICE", raising=False)
    sd = _FakeSd(
        default_ok=False,
        devices=[
            {"name": "speaker", "max_input_channels": 0},
            {"name": "mic", "max_input_channels": 2},
        ],
    )
    assert _resolve_input_device(sd) == 1


def test_resolve_device_no_input_devices_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.orchestration.voice_input import _resolve_input_device

    monkeypatch.delenv("ZERO_ASR_INPUT_DEVICE", raising=False)
    sd = _FakeSd(default_ok=False, devices=[{"name": "speaker", "max_input_channels": 0}])
    with pytest.raises(RuntimeError, match="录音设备"):
        _resolve_input_device(sd)


def test_resolve_device_env_index_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.orchestration.voice_input import _resolve_input_device

    sd = _FakeSd(
        default_ok=True,
        devices=[
            {"name": "speaker", "max_input_channels": 0},
            {"name": "mix", "max_input_channels": 2},
        ],
    )
    monkeypatch.setenv("ZERO_ASR_INPUT_DEVICE", "1")
    assert _resolve_input_device(sd) == 1
    monkeypatch.setenv("ZERO_ASR_INPUT_DEVICE", "0")
    with pytest.raises(RuntimeError, match="输入通道"):
        _resolve_input_device(sd)


def test_record_normalizes_peak_before_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    """弱信号峰值归一到 0.9 再进转写（环回实测：不归一 whisper 对弱音频幻听 prompt）。"""
    np = pytest.importorskip("numpy")
    received: list[Any] = []

    def spy_transcribe(audio: Any, language: str, **kwargs: Any) -> tuple[list[Any], Any]:
        received.append(audio)
        return [SimpleNamespace(text="好")], SimpleNamespace(language=language)

    vi = VoiceInput(model=SimpleNamespace(transcribe=spy_transcribe))
    quiet = np.array([0.001, -0.002, 0.0015], dtype=np.float32)
    monkeypatch.setattr(VoiceInput, "_capture_until_enter", lambda self: quiet)
    assert vi.record_until_enter() == "好"
    assert abs(float(np.max(np.abs(received[0]))) - 0.9) < 1e-6


def test_record_silence_gate_skips_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    """纯静音（峰值低于门限）不进转写直接空串。"""
    np = pytest.importorskip("numpy")

    def boom(audio: Any, language: str, **kwargs: Any) -> Any:
        raise AssertionError("静音不应进转写")

    vi = VoiceInput(model=SimpleNamespace(transcribe=boom))
    silence = np.full(100, 1e-6, dtype=np.float32)
    monkeypatch.setattr(VoiceInput, "_capture_until_enter", lambda self: silence)
    assert vi.record_until_enter() == ""


def test_record_until_enter_empty_capture_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """采集为空（没录到任何块）时不调模型直接返回空串。"""
    called: list[bool] = []

    def boom_transcribe(audio: Any, language: str) -> Any:
        called.append(True)
        raise AssertionError("空音频不应进转写")

    vi = VoiceInput(model=SimpleNamespace(transcribe=boom_transcribe))
    monkeypatch.setattr(VoiceInput, "_capture_until_enter", lambda self: None)
    assert vi.record_until_enter() == ""
    assert called == []
