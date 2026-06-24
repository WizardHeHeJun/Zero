"""Mood 管线集成（A.7）：心境带来跨刺激的历史依赖/滞后；默认关零回归。"""

from __future__ import annotations

from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.orchestration.state import Stimulus
from src.storage.checkpointer import build_checkpointer


def _graph():
    return build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())


async def test_mood_history_bends_present() -> None:
    graph = _graph()
    neg = Stimulus(name="loss", goal_congruence=-0.9, intensity=1.0)
    mild = Stimulus(name="mild", goal_congruence=0.2, intensity=0.3)
    # 同一 thread 连灌负面 → 心境累积进负盆
    for _ in range(6):
        await graph.ainvoke(
            {"stimulus": neg, "mood_enabled": True, "rng_seed": 0},
            config={"configurable": {"thread_id": "moody"}},
        )
    moody = await graph.ainvoke(
        {"stimulus": mild, "mood_enabled": True, "rng_seed": 0},
        config={"configurable": {"thread_id": "moody"}},
    )
    # 全新 thread：同样温和刺激、无负面历史
    fresh = await graph.ainvoke(
        {"stimulus": mild, "mood_enabled": True, "rng_seed": 0},
        config={"configurable": {"thread_id": "fresh"}},
    )
    # 有负面过去 → 当前情绪被心境拽得更负（同一 rng_seed，差异纯由历史心境引入）
    assert moody["affect_sample"][0] < fresh["affect_sample"][0]


async def test_mood_disabled_carries_no_mood() -> None:
    graph = _graph()
    neg = Stimulus(name="loss", goal_congruence=-0.9, intensity=1.0)
    result = await graph.ainvoke(
        {"stimulus": neg, "rng_seed": 0},
        config={"configurable": {"thread_id": "no-mood"}},
    )
    # 默认关：不产出 mood（运行态保持 None），v1 路径不受影响
    assert result.get("mood") is None
