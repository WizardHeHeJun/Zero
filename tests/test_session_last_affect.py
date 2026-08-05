"""`ConversationSession.last_affect()`：动作层的只读接入点。

三条性质：step 前有安全默认、step 后拿到本轮读数、**调用它不推进图**。
最后一条是设计的核心约束——动作层按自己节奏拉取，绝不能顺带跑一轮引擎。
"""

from __future__ import annotations

import pytest

from src.orchestration.runner import ConversationSession
from src.orchestration.state import Stimulus


@pytest.mark.asyncio
async def test_last_affect_before_any_step() -> None:
    """尚未 step → (None, None, 默认 leak)，调用方据此走 idle 基线，而不是崩。"""
    session = ConversationSession(thread_id="motion-t1")
    try:
        sample, regulated, leak = session.last_affect()
        assert sample is None
        assert regulated is None
        assert leak == 1.0
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_last_affect_reflects_latest_step() -> None:
    """step 后能拿到本轮的 e\\*，且与 step 返回体里的读数一致（同一来源，不会分叉）。"""
    session = ConversationSession(thread_id="motion-t2")
    try:
        entry = await session.step(Stimulus(name="s1", goal_congruence=-0.6, novelty=0.5))
        sample, _, _ = session.last_affect()
        assert sample is not None
        assert sample == pytest.approx(tuple(entry["valence_arousal"]))
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_last_affect_does_not_advance_graph() -> None:
    """⚠ 只读：连续调用 `last_affect()` 不改变任何状态，也不产生新的一轮。

    撤掉「只读」这条（比如实现成内部再 ainvoke 一次）会红。
    """
    session = ConversationSession(thread_id="motion-t3")
    try:
        await session.step(Stimulus(name="s1", goal_congruence=0.4, novelty=0.2))
        first = session.last_affect()
        for _ in range(5):
            assert session.last_affect() == first
        # 图确实没被推进：再 step 一次才会变
        await session.step(Stimulus(name="s2", goal_congruence=-0.8, novelty=0.9))
        assert session.last_affect() != first
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_last_affect_is_not_persisted_across_sessions() -> None:
    """缓存是实例属性、不进 checkpointer ⇒ 同 thread 新建会话应回到初始值。

    这条钉死「不进运行态持久化」的设计——若有人把它塞进 AffectState/checkpoint，这里会红。
    """
    session = ConversationSession(thread_id="motion-t4")
    try:
        await session.step(Stimulus(name="s1", goal_congruence=0.5, novelty=0.5))
        assert session.last_affect()[0] is not None
    finally:
        await session.aclose()

    fresh = ConversationSession(thread_id="motion-t4")
    try:
        assert fresh.last_affect()[0] is None
    finally:
        await fresh.aclose()
