"""ChatDriver（对话核心从临时入口 main.py 迁出到编排层）单测：一轮 step 串联 + 两时间尺度 + 落盘。

用假 session 注入固定 e*、lm=None 走词典回退，把噪声 gauss monkeypatch 成 0 求确定性，
验证「评价→引擎→两时间尺度情绪→生成→落盘」一轮链路与状态推进（行为零回归的护栏）。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.orchestration.chat_driver import ChatDriver, ChatTurn
from src.storage.conversation_log import ConversationLog


class _FakeSession:
    """假 ConversationSession：step 返回固定 e*，并记录被喂入的 stim.text（gist 依赖）。"""

    def __init__(self, ev: tuple[float, float]) -> None:
        self.ev = ev
        self.seen_text: str | None = None

    async def step(self, stim: Any) -> dict[str, Any]:
        self.seen_text = stim.text
        return {"valence_arousal": self.ev, "recalled_context": []}


def _make_driver(
    log: ConversationLog,
    session: Any,
    *,
    lm: Any = None,
    attitude: tuple[float, float] = (0.0, 0.0),
) -> ChatDriver:
    return ChatDriver(
        thread="t", lm=lm, log=log, session=session, history=[], attitude=attitude, mode="test"
    )


async def test_step_returns_turn_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    """一轮 step：返回 ChatTurn、词典回退回复含情绪标签、user+assistant+态度全部落盘。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    session = _FakeSession((0.6, 0.4))
    driver = _make_driver(log, session)
    turn = await driver.step("我很开心")
    assert isinstance(turn, ChatTurn)
    assert turn.emotion_label in turn.reply  # 词典回退模板带情绪标签
    assert session.seen_text == "我很开心"  # 本句 text 喂进引擎（gist 依赖）
    assert [r["content"] for r in log.recent("t", 10)] == ["我很开心", turn.reply]
    assert log.load_feeling("t") == pytest.approx(turn.attitude)  # 只持久化态度
    log.close()


async def test_two_timescale_both_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """正向 e* 下，慢变 attitude 与快变 emotion 都从初值发生更新（两时间尺度都动）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    driver = _make_driver(log, _FakeSession((0.8, 0.3)))
    turn = await driver.step("好棒")
    assert turn.attitude != (0.0, 0.0)
    assert turn.emotion != (0.0, 0.0)
    assert turn.attitude == driver.attitude  # 持有状态与返回同步
    log.close()


async def test_attitude_prior_is_pre_step_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """attitude_prior 是进入本轮前的快照（attitude_step 在 step 之后才更新）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    driver = _make_driver(log, _FakeSession((0.8, 0.3)), attitude=(0.25, 0.1))
    turn = await driver.step("hi")
    assert turn.attitude_prior == 0.25
    assert turn.attitude[0] != 0.25  # 已被 attitude_step 推进
    log.close()
