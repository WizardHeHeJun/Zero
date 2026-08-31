"""按句切分流水（ZERO_TTS_SENTENCE_SPLIT，2026-08-31 计划④）。

覆盖：切分纯函数（终止符/括号保护/省略号/短句合并/换行/无终止符）·
门关默认整句零回归锚点 · 门开逐段投递序 · 分段失败跳过不断流 · 工厂 env 接线。
夹具复用 test_expression_speech 的 FakeTransport/_fake_synth 模式。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.expression_out.base import ExpressionFrame
from src.expression_out.speech import TtsSpeechSink, build_speech_sink, split_sentences
from tests.test_expression_speech import FakeTransport, _fake_synth, _wav

FPS = 20.0


# ── split_sentences 纯函数 ───────────────────────────────────────────────────


def test_split_basic_three_sentences() -> None:
    text = "今天天气真是好得不得了啊。我们一起出去公园走一走吧？回头再回来慢慢吃晚饭好不好。"
    assert split_sentences(text) == [
        "今天天气真是好得不得了啊。",
        "我们一起出去公园走一走吧？",
        "回头再回来慢慢吃晚饭好不好。",
    ]


def test_split_merges_short_segments_forward() -> None:
    """「嗯。」级碎片贪心向后合并到 ≥ min_chars——防碎片合成开销与颗粒停顿。"""
    assert split_sentences("嗯。好的呀。那我们现在就赶紧出发吧！") == [
        "嗯。好的呀。那我们现在就赶紧出发吧！"
    ]


def test_split_protects_quotes_and_brackets() -> None:
    """引号/括号内的终止符不切（切开会把引语从中间断开）。"""
    text = "他说「今天不去了。改天吧。」然后就把电话挂断了。"
    assert split_sentences(text) == [text]


def test_split_keeps_consecutive_terminators_together() -> None:
    """「……」的第二个 … 是纯终止符残段，并回上一段而非独立成段。"""
    segments = split_sentences("让我好好想一想这个问题……应该是可以的吧我觉得没什么问题。")
    assert len(segments) == 2
    assert segments[0].endswith("……")


def test_split_short_tail_merges_backward() -> None:
    segments = split_sentences("今天我们把这件事情聊清楚。嗯")
    assert segments == ["今天我们把这件事情聊清楚。嗯"]


def test_split_newline_is_boundary() -> None:
    segments = split_sentences("第一行的内容说完了没有标点\n第二行也说了不少的内容呢")
    assert len(segments) == 2


def test_split_no_terminator_returns_whole() -> None:
    assert split_sentences("这句话没有结束符") == ["这句话没有结束符"]


def test_split_empty() -> None:
    assert split_sentences("") == []


def test_split_all_terminators_degenerates_to_single() -> None:
    """全终止符文本（「。。。」）不崩、退化为单段。"""
    assert split_sentences("。。。") == ["。。。"]


# ── sink 集成 ────────────────────────────────────────────────────────────────

_THREE = "今天天气真是好得不得了啊。我们一起出去公园走一走吧？回头再回来慢慢吃晚饭好不好。"


def _sink(transport: Any, tmp_path: Path, *, split: bool) -> TtsSpeechSink:
    return TtsSpeechSink(
        transport=transport,
        server_url="http://127.0.0.1:5000/voice",
        speaker="test",
        fps=FPS,
        wav_dir=tmp_path / "tts",
        sentence_split=split,
    )


async def _wait_calls(transport: FakeTransport, n: int, budget_s: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + budget_s
    while len(transport.calls) < n:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"speech_play 调用数 {len(transport.calls)} < 期望 {n}")
        await asyncio.sleep(0.02)


async def test_default_off_single_synth_zero_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门关（默认）锚点：多句回复仍整句一次合成、一次 speech_play——v1 逐字行为。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, split=False)
    spoken: list[str] = []
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(), spoken))
    await sink.connect()
    await sink.emit(ExpressionFrame(emotion=(0.0, 0.0), emotion_label="平静", reply=_THREE))
    await _wait_calls(transport, 1)
    await asyncio.sleep(0.05)
    assert spoken == [_THREE]
    assert len(transport.calls) == 1
    await sink.aclose()


async def test_split_on_delivers_segments_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门开：逐段合成→逐段 speech_play，投递序=文本序，wav 路径逐段不同。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, split=True)
    spoken: list[str] = []
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(), spoken))
    await sink.connect()
    await sink.emit(ExpressionFrame(emotion=(0.0, 0.0), emotion_label="平静", reply=_THREE))
    await _wait_calls(transport, 3)
    assert spoken == split_sentences(_THREE)
    wav_paths = [args["wav_path"] for name, args in transport.calls if name == "speech_play"]
    assert len(wav_paths) == 3
    assert len(set(wav_paths)) == 3
    await sink.aclose()


async def test_split_segment_failure_skips_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第 2 段合成失败：跳过该段，第 1/3 段仍投递（分段失败不断流，对话不受影响）。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path, split=True)
    segments = split_sentences(_THREE)
    attempted: list[str] = []

    async def flaky(self: TtsSpeechSink, text: str) -> bytes:
        attempted.append(text)
        if text == segments[1]:
            raise RuntimeError("TTS 服务抖了一下")
        return _wav()

    monkeypatch.setattr(TtsSpeechSink, "_synthesize", flaky)
    await sink.connect()
    await sink.emit(ExpressionFrame(emotion=(0.0, 0.0), emotion_label="平静", reply=_THREE))
    await _wait_calls(transport, 2)
    await asyncio.sleep(0.05)
    assert attempted == segments
    assert len(transport.calls) == 2
    await sink.aclose()


def test_factory_env_wires_sentence_split(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("httpx")
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SERVER_URL", "http://127.0.0.1:5000/voice")
    monkeypatch.setenv("ZERO_TTS_SPEAKER", "test")
    monkeypatch.setenv("ZERO_TTS_SENTENCE_SPLIT", "true")
    sink = build_speech_sink(transport=FakeTransport())  # type: ignore[arg-type]
    assert sink is not None and sink.sentence_split is True
    monkeypatch.delenv("ZERO_TTS_SENTENCE_SPLIT")
    sink2 = build_speech_sink(transport=FakeTransport())  # type: ignore[arg-type]
    assert sink2 is not None and sink2.sentence_split is False
