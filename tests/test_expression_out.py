"""表现层出口测试（`src/expression_out`）。

核心断言：
1. 零回归——不配 sink 时对话逐字不变、`_express` 是 no-op。
2. 红线「表现失败不扳倒对话」——sink 抛异常时 `step` 照常返回完整 ChatTurn。
3. 帧内容正确——情绪/回复/regulated 如实传递（表现端据此驱动）。
4. VtsSink 的降级姿态——渲染端起不来时 `connect()` 返回 False 而非抛，`emit` 静默 no-op。
5. 协议一致性——VtsSink 结构上满足 `ExpressionSink`（runtime_checkable）。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.expression_out.base import ExpressionFrame, ExpressionSink
from src.expression_out.factory import build_expression_sinks
from src.expression_out.vts import VtsSink


class RecordingSink:
    """记录收到的帧；可选地在 emit 时抛异常（模拟渲染端故障）。

    刻意**不做任何连接**——代表"无需连接的表现形式"（纯文字标签/写文件那类），
    用来钉住协议必须能容纳这种实现（`connect` 直接返回 True）。
    """

    def __init__(self, *, boom: bool = False) -> None:
        self.frames: list[ExpressionFrame] = []
        self.boom = boom
        self.closed = False
        self.connected = False

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def emit(self, frame: ExpressionFrame) -> None:
        self.frames.append(frame)
        if self.boom:
            raise RuntimeError("渲染端炸了")

    async def aclose(self) -> None:
        self.closed = True


# ── 协议与数据结构 ────────────────────────────────────────────────────────────


def test_recording_sink_satisfies_protocol() -> None:
    assert isinstance(RecordingSink(), ExpressionSink)


def test_vts_sink_satisfies_protocol() -> None:
    """VtsSink 未连接也应结构满足协议（协议是结构性的，不依赖运行状态）。"""
    assert isinstance(VtsSink(), ExpressionSink)


def test_frame_is_frozen() -> None:
    """帧是不可变的：表现端拿到后不得改写它再传给下一个 sink。"""
    frame = ExpressionFrame(emotion=(0.1, 0.2), emotion_label="平静", reply="嗯")
    with pytest.raises(Exception):  # noqa: B017 —— dataclass frozen 抛 FrozenInstanceError
        frame.emotion = (0.9, 0.9)  # type: ignore[misc]


# ── 工厂：默认关 ─────────────────────────────────────────────────────────────


def test_factory_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配任何 sink env → 空列表 → 不表现（零回归的第一重保证）。"""
    monkeypatch.delenv("ZERO_VTS_SINK", raising=False)
    assert build_expression_sinks() == []


def test_factory_builds_vts_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_VTS_SINK", "true")
    sinks = build_expression_sinks()
    assert len(sinks) == 1
    assert isinstance(sinks[0], VtsSink)


# ── ChatDriver 接线：零回归 + 失败不打断 ─────────────────────────────────────


async def _driver(sinks: list[Any]) -> Any:
    """构造一个不依赖 LLM/外部后端的 ChatDriver（模板回退路径）。"""
    from src.memory.client import MemoryClient
    from src.orchestration.chat_driver import ChatDriver
    from src.orchestration.runner import ConversationSession
    from src.storage.conversation_log import ConversationLog

    return ChatDriver(
        thread="t-express",
        lm=None,  # 无 LLM → 模板回复，测试不依赖网络
        log=ConversationLog(":memory:"),
        session=ConversationSession(thread_id="t-express", memory=MemoryClient()),
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        expression_sinks=sinks,
    )


async def test_no_sinks_is_noop() -> None:
    """不配 sink：step 正常返回，_express 不做任何事（零回归）。"""
    driver = await _driver([])
    turn = await driver.step("你好")
    assert turn.reply
    assert driver.expression_sinks == []


async def test_frame_carries_emotion_and_reply() -> None:
    """帧如实携带本轮情绪/标签/回复——表现端全靠它驱动。"""
    sink = RecordingSink()
    driver = await _driver([sink])
    turn = await driver.step("今天真开心")
    assert len(sink.frames) == 1
    frame = sink.frames[0]
    assert frame.emotion == turn.emotion
    assert frame.emotion_label == turn.emotion_label
    assert frame.reply == turn.reply


async def test_sink_failure_does_not_break_conversation() -> None:
    """🛑 红线：表现端抛异常，对话必须照常完成（下游故障不回灌上游）。

    变异验证：把 `_express` 里的 try/except 去掉，本用例会当场红。
    """
    boom = RecordingSink(boom=True)
    quiet = RecordingSink()
    driver = await _driver([boom, quiet])
    turn = await driver.step("你好啊")
    assert turn.reply  # 对话没被打断
    assert len(boom.frames) == 1
    assert len(quiet.frames) == 1  # 前一个 sink 炸了不影响后一个


async def test_multiple_turns_emit_each() -> None:
    sink = RecordingSink()
    driver = await _driver([sink])
    await driver.step("第一句")
    await driver.step("第二句")
    assert len(sink.frames) == 2


# ── VtsSink 降级姿态 ─────────────────────────────────────────────────────────


async def test_vts_connect_failure_returns_false(tmp_path: Any) -> None:
    """渲染端起不来时 connect() 返回 False 而不是抛——调用方可继续纯对话。

    指向一个不存在的仓路径，worker 必然起不来/连不上。
    """
    sink = VtsSink(mcp_repo=tmp_path / "no-such-repo", token_file=tmp_path / "tok")
    assert await sink.connect() is False
    await sink.aclose()  # 幂等：没连上也能安全关


async def test_vts_emit_without_connect_short_circuits() -> None:
    """未连接时 emit **提前短路**——不是"抛了被自己吞掉"。

    区分力（code-reviewer 2026-08-11 指出旧版验不出东西）：旧断言只看"不抛异常"，
    而删掉 `self.proc is None` 那道早退护栏后，下游 `_rpc` 抛的 RuntimeError 同样会被
    emit 自己的 try/except 吞掉 ⇒ 测试照绿。现在直接监视 `_rpc`：未连接时它一次都不该被调。
    """
    sink = VtsSink()
    calls: list[dict[str, Any]] = []

    async def spy(msg: dict[str, Any]) -> dict[str, Any]:
        calls.append(msg)
        return {"ok": True}

    sink._rpc = spy  # type: ignore[method-assign]
    await sink.emit(ExpressionFrame(emotion=(0.5, 0.5), emotion_label="欣喜", reply="是这样"))
    assert calls == []  # 护栏被删则这里会收到 behavior 指令
    await sink.aclose()


async def test_sink_without_connection_step_satisfies_protocol() -> None:
    """无需连接的表现形式也能满足协议并被入口统一调用（协议含 connect 的理由）。

    此前 connect 是 VtsSink 独有的扩展方法，入口靠鸭子类型硬调——加一个没有该方法的
    实现就 AttributeError（mypy main.py 实证）。现在协议要求它，本用例钉住这条。
    """
    sink = RecordingSink()
    assert isinstance(sink, ExpressionSink)
    assert await sink.connect() is True
    assert sink.connected


def test_vts_intents_reuse_closed_vocabulary() -> None:
    """离散行为抽取复用 12 词闭集：映射不进闭集的动作宣称照旧被丢弃。"""
    sink = VtsSink()
    assert [i.name for i in sink._intents("（点了点头）嗯，是这样。")] == ["nod"]
    assert sink._intents("（我帮你把灯关了）好嘞。") == []
