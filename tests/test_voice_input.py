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
    def transcribe(audio: Any, language: str) -> tuple[list[Any], Any]:
        segments = [SimpleNamespace(text=t) for t in texts]
        return segments, SimpleNamespace(language=language)

    return SimpleNamespace(transcribe=transcribe)


def test_transcribe_joins_segments_and_strips() -> None:
    vi = VoiceInput(model=_fake_model([" 今天天气", "不错。 "]))
    assert vi.transcribe(object()) == "今天天气不错。"


def test_transcribe_empty_segments_returns_empty() -> None:
    vi = VoiceInput(model=_fake_model([]))
    assert vi.transcribe(object()) == ""


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
