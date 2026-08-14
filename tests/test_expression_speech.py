"""语音表现 sink 测试（`src/expression_out/speech.py`，speech-output v1）。

全程 fake transport + fake 合成（不依赖 httpx/Bert-VITS2/渲染端）：

1. 协议一致性 + emit 投队列即返、worker 串行投递 `speech_play`。
2. 降级三分支——合成失败 / `speech_play` 报错 / 未连接，均不抛、不影响后续。
3. 舞台说明不出声（strip 后合成）；全括注轮次跳过。
4. 韵律流留口的契约形状（PRD G4）。
5. `build_speech_sink` fail-fast（URL/speaker/httpx 三缺口）与工厂共享 transport。
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import sys
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.expression_out.base import ExpressionFrame, ExpressionSink
from src.expression_out.lipsync import MOUTH_PARAMS
from src.expression_out.speech import ProsodyFrame, TtsSpeechSink, build_speech_sink
from src.expression_out.transport import VtsTransport

FPS = 20.0


def _wav(seconds: float = 0.3, amplitude: float = 0.7, rate: int = 44100) -> bytes:
    samples = array("h")
    for i in range(int(seconds * rate)):
        samples.append(int(amplitude * 32000 * math.sin(2 * math.pi * 220.0 * i / rate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _reply_ok(duration_ms: float = 0.0) -> SimpleNamespace:
    text = json.dumps({"accepted": True, "duration_ms": duration_ms})
    return SimpleNamespace(isError=False, content=[SimpleNamespace(text=text)])


def _reply_error(msg: str = "[zero_mcp:no-audio-device] 播放设备不可用") -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=msg)])


class FakeTransport:
    """替身 transport：记录 call_tool；connect 可配成失败。"""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.session: Any = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.reply: Any = _reply_ok()
        self.closed = 0

    async def connect(self) -> bool:
        if not self.ok:
            return False
        self.session = object()
        return True

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))
        return self.reply

    async def aclose(self) -> None:
        self.closed += 1
        self.session = None


def _sink(transport: Any, tmp_path: Path) -> TtsSpeechSink:
    return TtsSpeechSink(
        transport=transport,
        server_url="http://127.0.0.1:5000/voice",
        speaker="test",
        fps=FPS,
        wav_dir=tmp_path / "tts",
    )


def _frame(reply: str) -> ExpressionFrame:
    return ExpressionFrame(emotion=(0.2, 0.1), emotion_label="平静", reply=reply)


async def _drain(sink: TtsSpeechSink) -> None:
    """等 worker 消化完队列（fake 路径应在亚秒级完成）。"""
    for _ in range(500):
        if sink.queue.empty():
            await asyncio.sleep(0.02)
            if sink.queue.empty():
                return
        await asyncio.sleep(0.01)
    raise AssertionError("worker 没有在预算时间内消化完队列")


async def _spoken(sink: TtsSpeechSink, transport: FakeTransport, reply: str) -> None:
    """emit 一轮并等到投递完成。"""
    before = len(transport.calls)
    await sink.emit(_frame(reply))
    for _ in range(500):
        if len(transport.calls) > before or sink.queue.empty():
            break
        await asyncio.sleep(0.01)
    await _drain(sink)


# ── 协议与主链路 ─────────────────────────────────────────────────────────────


def test_speech_sink_satisfies_protocol(tmp_path: Path) -> None:
    assert isinstance(_sink(FakeTransport(), tmp_path), ExpressionSink)


async def test_emit_enqueues_and_worker_plays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """emit 即返；worker 完成 合成→口型→speech_play，payload 形状符合跨仓规范。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav()))
    assert await sink.connect() is True
    await _spoken(sink, transport, "你好呀")

    plays = [c for c in transport.calls if c[0] == "speech_play"]
    assert len(plays) == 1
    args = plays[0][1]
    assert Path(args["wav_path"]).exists(), "wav 应已落盘且路径可读（同机契约）"
    assert args["fps"] == FPS
    assert args["mouth_track"], "口型轨迹不该为空"
    for kf in args["mouth_track"]:
        assert set(kf["params"]) <= set(MOUTH_PARAMS), "投递 payload 越出嘴部集合"
    await sink.aclose()


async def test_emit_before_connect_is_noop(tmp_path: Path) -> None:
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    await sink.emit(_frame("你好"))
    assert sink.queue.empty(), "未连接时 emit 应短路，不积压队列"
    assert transport.calls == []
    await sink.aclose()  # 幂等：没连上也能安全关


async def test_connect_failure_returns_false(tmp_path: Path) -> None:
    sink = _sink(FakeTransport(ok=False), tmp_path)
    assert await sink.connect() is False
    assert sink.worker_task is None, "连接失败不该起 worker"


# ── 舞台说明与空文本 ─────────────────────────────────────────────────────────


def _fake_synth(wav_bytes: bytes, spoken: list[str] | None = None, boom: bool = False) -> Any:
    async def synth(self: TtsSpeechSink, text: str) -> bytes:
        if spoken is not None:
            spoken.append(text)
        if boom:
            raise RuntimeError("TTS 服务连不上")
        return wav_bytes

    return synth


async def test_stage_directions_are_not_spoken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """「（笑）」类括注绝不进合成文本——否则会被数字人读出来。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    spoken: list[str] = []
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(), spoken))
    await sink.connect()
    await _spoken(sink, transport, "（轻轻笑了笑）今天过得怎么样？")
    assert spoken == ["今天过得怎么样？"]
    await sink.aclose()


async def test_pure_stage_direction_reply_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    spoken: list[str] = []
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(), spoken))
    await sink.connect()
    await _spoken(sink, transport, "（点了点头）")
    assert spoken == []
    assert [c for c in transport.calls if c[0] == "speech_play"] == []
    await sink.aclose()


async def test_known_miss_pure_bracket_normal_speech_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KNOWN_MISS 钉住：「（好的）」全括号**普通答话**也被跳过（判据是结构非语义）。

    这是已声明的可接受误伤（speech.py `_PURE_BRACKET_RE` 蕴含论证）；哪天判据能
    区分舞台说明与括号正文，本用例会红——届时更新蕴含论证与本用例，别当 bug 顺手修。
    """
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    spoken: list[str] = []
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(), spoken))
    await sink.connect()
    await _spoken(sink, transport, "（好的）")
    assert spoken == []
    await sink.aclose()


# ── _synthesize 本体：hiyoriUI 契约（httpx.MockTransport，不走真网络）────────


async def test_synthesize_passes_hiyori_params_and_accepts_wav(tmp_path: Path) -> None:
    """契约字段（model_id/speaker_name/language）如实上送，RIFF wav 原样返回。"""
    httpx = pytest.importorskip("httpx")
    sink = _sink(FakeTransport(), tmp_path)
    seen: dict[str, str] = {}

    def handler(request: Any) -> Any:
        seen.update({k: v for k, v in request.url.params.items()})
        return httpx.Response(200, content=_wav())

    sink.http_transport = httpx.MockTransport(handler)
    body = await sink._synthesize("你好")
    assert body.startswith(b"RIFF")
    assert seen["text"] == "你好"
    assert seen["model_id"] == "0"
    assert seen["speaker_name"] == "test"
    assert seen["language"] == "ZH"


async def test_synthesize_raises_on_hiyori_business_error_json(tmp_path: Path) -> None:
    """hiyoriUI 业务错误是 **HTTP 200 + JSON**——只看状态码会把 JSON 当 wav 吞进口型。

    变异判别力：把 `_synthesize` 里的 RIFF 头校验删掉，本用例当场红。
    """
    httpx = pytest.importorskip("httpx")
    sink = _sink(FakeTransport(), tmp_path)

    def handler(request: Any) -> Any:
        return httpx.Response(200, json={"status": 10, "detail": "模型model_id=0未加载"})

    sink.http_transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError, match="未加载"):
        await sink._synthesize("你好")


# ── 降级：任何失败不外抛、不影响后续轮次 ──────────────────────────────────────


async def test_synthesis_failure_degrades_and_next_turn_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    attempts: list[str] = []

    async def flaky(self: TtsSpeechSink, text: str) -> bytes:
        attempts.append(text)
        if len(attempts) == 1:
            raise RuntimeError("TTS 服务连不上")
        return _wav()

    monkeypatch.setattr(TtsSpeechSink, "_synthesize", flaky)
    await sink.connect()
    await _spoken(sink, transport, "第一句")
    await _spoken(sink, transport, "第二句")
    assert len(attempts) == 2, "第一句失败不该拖垮 worker"
    assert len([c for c in transport.calls if c[0] == "speech_play"]) == 1
    await sink.aclose()


async def test_speech_play_rejection_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """渲染端拒收（isError）：只记日志，不抛、不重试轰炸。"""
    transport = FakeTransport()
    transport.reply = _reply_error()
    sink = _sink(transport, tmp_path)
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav()))
    await sink.connect()
    await _spoken(sink, transport, "你好")
    assert len([c for c in transport.calls if c[0] == "speech_play"]) == 1
    await sink.aclose()


# ── 韵律流留口（PRD G4 契约）──────────────────────────────────────────────────


async def test_prosody_frames_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """形状锁定：t_ms 从 0 按帧距递增、energy∈[0,1]；返回副本不可污染内部。"""
    transport = FakeTransport()
    sink = _sink(transport, tmp_path)
    monkeypatch.setattr(TtsSpeechSink, "_synthesize", _fake_synth(_wav(seconds=0.5)))
    await sink.connect()
    await _spoken(sink, transport, "契约测试")
    frames = sink.prosody_frames()
    assert frames, "合成后应有韵律帧"
    assert all(isinstance(f, ProsodyFrame) for f in frames)
    assert frames[0].t_ms == 0
    assert all(b.t_ms - a.t_ms == 50 for a, b in zip(frames, frames[1:], strict=False))
    assert all(0.0 <= f.energy <= 1.0 for f in frames)
    frames.clear()
    assert sink.prosody_frames(), "prosody_frames 应返回副本"
    await sink.aclose()


# ── build_speech_sink：默认关 + fail-fast ────────────────────────────────────


def test_build_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_TTS_SINK", raising=False)
    assert build_speech_sink() is None


def test_build_fails_fast_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.delenv("ZERO_TTS_SERVER_URL", raising=False)
    with pytest.raises(RuntimeError, match="ZERO_TTS_SERVER_URL"):
        build_speech_sink()


def test_build_fails_fast_without_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SERVER_URL", "http://127.0.0.1:5000/voice")
    monkeypatch.delenv("ZERO_TTS_SPEAKER", raising=False)
    with pytest.raises(RuntimeError, match="ZERO_TTS_SPEAKER"):
        build_speech_sink()


def test_build_fails_fast_without_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """门开但缺 tts extra ⇒ 构造期报错（部署上悄悄没声音比报错糟）。"""
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SERVER_URL", "http://127.0.0.1:5000/voice")
    monkeypatch.setenv("ZERO_TTS_SPEAKER", "test")
    monkeypatch.setitem(sys.modules, "httpx", None)  # None ⇒ import httpx 抛 ImportError
    with pytest.raises(RuntimeError, match="tts"):
        build_speech_sink()


def test_build_returns_sink_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("httpx")
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SERVER_URL", "http://127.0.0.1:5000/voice")
    monkeypatch.setenv("ZERO_TTS_SPEAKER", "paimon")
    monkeypatch.setenv("ZERO_TTS_LANGUAGE", "ZH")
    sink = build_speech_sink()
    assert isinstance(sink, TtsSpeechSink)
    assert sink.speaker == "paimon"


# ── 工厂：连接共享 + 零回归 ───────────────────────────────────────────────────


def test_factory_shares_one_transport_across_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """皮套与语音同开时必须共用同一条 VtsTransport（VTS 参数注入单进程独占）。"""
    pytest.importorskip("httpx")
    from src.expression_out.factory import build_expression_sinks

    monkeypatch.setenv("ZERO_VTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SINK", "true")
    monkeypatch.setenv("ZERO_TTS_SERVER_URL", "http://127.0.0.1:5000/voice")
    monkeypatch.setenv("ZERO_TTS_SPEAKER", "test")
    sinks = build_expression_sinks()
    assert len(sinks) == 2
    transports = {id(s.transport) for s in sinks}
    assert len(transports) == 1, "两个 sink 拿到了不同 transport ⇒ 会开出第二条 VTS 连接"


def test_factory_default_off_includes_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.expression_out.factory import build_expression_sinks

    monkeypatch.delenv("ZERO_VTS_SINK", raising=False)
    monkeypatch.delenv("ZERO_TTS_SINK", raising=False)
    assert build_expression_sinks() == []


# ── transport 幂等语义（多 sink 共享的前提）──────────────────────────────────


async def test_transport_connect_is_idempotent_when_connected() -> None:
    transport = VtsTransport()
    transport.session = object()  # 模拟已连接
    assert await transport.connect() is True, "已连接的 connect 应直接复用，不再 spawn"


async def test_transport_aclose_is_idempotent() -> None:
    transport = VtsTransport()
    await transport.aclose()
    await transport.aclose()  # 未连接/重复关都不该抛
