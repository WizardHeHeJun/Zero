"""MemoryRecallAgent（记忆读闭环）单测：门控、空记忆 no-op、解析、appraisal 偏置、端到端闭环。"""

from __future__ import annotations

from src.agents.appraisal import AppraisalAgent
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.graph import build_graph
from src.orchestration.memory_recall import MemoryRecallAgent
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer


async def test_recall_noop_when_disabled() -> None:
    out = await MemoryRecallAgent(MemoryClient())(AffectState(recall_enabled=False))
    assert out == {}


async def test_recall_noop_on_empty_memory() -> None:
    state = AffectState(recall_enabled=True, user_id="u1")
    assert await MemoryRecallAgent(MemoryClient())(state) == {}


async def test_recall_reads_latest_disposition() -> None:
    mem = MemoryClient()
    await mem.write("disposition stimulus=loss value=-0.700", scope=Scope.USER, key="u1")
    out = await MemoryRecallAgent(mem)(AffectState(recall_enabled=True, user_id="u1"))
    assert abs(out["recalled_disposition"] - (-0.7)) < 1e-9


def test_appraisal_biased_by_negative_disposition() -> None:
    stim = Stimulus(name="x", goal_congruence=0.0, intensity=0.5)
    base = AppraisalAgent()(AffectState(stimulus=stim))
    biased = AppraisalAgent()(AffectState(stimulus=stim, recalled_disposition=-0.8))
    assert biased["prior_mu"][0] < base["prior_mu"][0]  # 负倾向拉低先验 valence
    assert biased["reward"] == base["reward"]  # reward 不受回灌影响（TD 通路不变）


async def test_recall_closes_loop_end_to_end() -> None:
    mem = MemoryClient()
    graph = build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), mem)
    neg = Stimulus(name="loss", goal_congruence=-0.9, intensity=1.0)
    # 第一次负面：Supervisor 在任务完成写 user 长期倾向
    await graph.ainvoke(
        {"stimulus": neg, "recall_enabled": True, "user_id": "u1", "rng_seed": 0},
        config={"configurable": {"thread_id": "r1"}},
    )
    # 第二次中性刺激：MemoryRecall 读到负倾向 → 偏置 appraisal（记忆被用上）
    r2 = await graph.ainvoke(
        {
            "stimulus": Stimulus(name="neutral", goal_congruence=0.0, intensity=0.5),
            "recall_enabled": True,
            "user_id": "u1",
            "rng_seed": 0,
        },
        config={"configurable": {"thread_id": "r1"}},
    )
    assert r2["recalled_disposition"] is not None
    assert r2["recalled_disposition"] < 0
