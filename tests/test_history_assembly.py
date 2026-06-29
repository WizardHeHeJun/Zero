"""D1+D2：history U 形装配 + 召回注入 system 条目单测（PR-4）。

覆盖：
  - _u_shape_history：k=0 退化 history[-n:]（零回归）；len<=n 不截；k=3,n=20 头3+尾17；
    k>=n 退化尾窗。
  - _inject_recalled_as_system：空召回原样返回（BLOCK-2 fallback）；importance<阈值不注入；
    importance>=阈值以 system 条目插头部；保序。
  - ChatDriver.step 集成：高 importance 召回进 converse 的 history 头部为 system 条目；
    无召回时不注入（零回归），默认 K=3 生效。

纯数据/fake lm+session，不调真 LLM。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.memory.types import Fact, Scope
from src.orchestration.chat_driver import (
    ChatDriver,
    _inject_recalled_as_system,
    _u_shape_history,
)
from src.storage.conversation_log import ConversationLog


def _msgs(n: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(i)} for i in range(n)]


def _fact(content: str) -> Fact:
    return Fact(content=content, scope=Scope.USER, valid_at=datetime.now(UTC))


# --------------------------------------------------------------------------- #
# _u_shape_history
# --------------------------------------------------------------------------- #


def test_u_shape_k0_degrades_to_tail() -> None:
    """K=0 → history[-n:]（逐字节等价原行为，零回归开关）。"""
    h = _msgs(40)
    assert _u_shape_history(h, 0, 20) == h[-20:]


def test_u_shape_short_history_not_truncated() -> None:
    """len<=n → 全量返回，不截断。"""
    h = _msgs(10)
    assert _u_shape_history(h, 3, 20) == h


def test_u_shape_k3_n20_head_plus_tail() -> None:
    """40 条、K=3、N=20 → 头 3 + 尾 17，无重叠。"""
    h = _msgs(40)
    window = _u_shape_history(h, 3, 20)
    assert len(window) == 20
    assert window[:3] == h[:3], "首因端=最早 3 条"
    assert window[3:] == h[-17:], "近因端=最近 17 条"


def test_u_shape_k_ge_n_degrades() -> None:
    """K>=N → 退化为纯尾窗（不出现负切片越界）。"""
    h = _msgs(40)
    assert _u_shape_history(h, 20, 20) == h[-20:]


# --------------------------------------------------------------------------- #
# _inject_recalled_as_system
# --------------------------------------------------------------------------- #


def test_inject_empty_returns_window_unchanged() -> None:
    """空召回 → 原样返回 window（BLOCK-2 fallback，零回归）。"""
    window = _msgs(3)
    assert _inject_recalled_as_system(window, [], 0.5) is window


def test_inject_below_threshold_skipped() -> None:
    """importance < inject_min → 不生成 system 条目。"""
    window = _msgs(2)
    out = _inject_recalled_as_system(window, [_fact("旧 | precision=0.10")], 0.5)
    assert out == window
    assert all(m["role"] != "system" for m in out)


def test_inject_above_threshold_front_system() -> None:
    """importance >= inject_min → system 条目插 window 头部。"""
    window = _msgs(2)
    out = _inject_recalled_as_system(window, [_fact("要事 | precision=0.90")], 0.5)
    assert out[0]["role"] == "system"
    assert "要事" in out[0]["content"]
    assert out[1:] == window, "原 window 顺序不变，仅头部前插"


def test_inject_preserves_ranked_order() -> None:
    """多条达标 episode 按入参（已重排）顺序插入头部。"""
    window = _msgs(1)
    facts = [_fact("第一 | precision=0.90"), _fact("第二 | precision=0.80")]
    out = _inject_recalled_as_system(window, facts, 0.5)
    assert "第一" in out[0]["content"]
    assert "第二" in out[1]["content"]


# --------------------------------------------------------------------------- #
# ChatDriver.step 集成
# --------------------------------------------------------------------------- #


class _RecordingLM:
    """假 lm：记录传入 converse 的 history（验证装配结果），appraise 返回固定 (v,a)。"""

    def __init__(self) -> None:
        self.seen_history: list[dict[str, str]] | None = None

    async def appraise_text(self, text: str) -> tuple[float, float]:
        return (0.5, 0.5)

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
    ) -> str:
        self.seen_history = history
        return "好的"


class _FactSession:
    """假 session：step 返回固定 e* + 指定 recalled_facts。"""

    def __init__(self, facts: list[Fact]) -> None:
        self.facts = facts

    async def step(self, stim: Any) -> dict[str, Any]:
        return {
            "valence_arousal": (0.5, 0.5),
            "recalled_context": [f.content for f in self.facts],
            "recalled_facts": self.facts,
        }


def _driver(lm: Any, session: Any) -> ChatDriver:
    return ChatDriver(
        thread="t",
        lm=lm,
        log=ConversationLog(":memory:"),
        session=session,
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
    )


async def test_step_injects_high_importance_system_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """高 importance 召回进 converse 的 history 头部为 system 条目；低 importance 不进。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    monkeypatch.delenv("ZERO_RECALL_INJECT_MIN", raising=False)  # 默认 0.5
    lm = _RecordingLM()
    facts = [_fact("要事 | precision=0.90"), _fact("琐事 | precision=0.10")]
    driver = _driver(lm, _FactSession(facts))
    await driver.step("你好")
    assert lm.seen_history is not None
    system_msgs = [m for m in lm.seen_history if m["role"] == "system"]
    assert len(system_msgs) == 1, "仅高 importance(0.90>=0.5) 进 system，低的(0.10)被挡"
    assert "要事" in system_msgs[0]["content"]
    driver.log.close()


async def test_step_no_recall_zero_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """无召回 → 不注入 system 条目，converse 仅收到 U 形 history（零回归）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    lm = _RecordingLM()
    driver = _driver(lm, _FactSession([]))
    await driver.step("你好")
    assert lm.seen_history is not None
    assert all(m["role"] != "system" for m in lm.seen_history)
    assert lm.seen_history[-1]["content"] == "你好"
    driver.log.close()
