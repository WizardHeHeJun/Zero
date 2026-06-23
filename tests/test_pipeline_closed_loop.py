"""T6.3 闭环：跑序列得 (v,a) 轨迹 + 中间量齐全；重复同一刺激方差有界不发散。（G1）"""

from __future__ import annotations

import statistics

from src.agents.affect_math import MAX_SAMPLE_SIGMA
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES, run
from src.orchestration.state import Stimulus
from src.storage.checkpointer import build_checkpointer


async def test_closed_loop_produces_trajectory_with_intermediates() -> None:
    traj = await run(
        [Stimulus(name="win", goal_congruence=0.8, intensity=0.9)],
        thread_id="cl1",
        rng_seed=1,
    )
    step = traj[0]
    assert step["valence_arousal"] is not None
    for key in ("prior_mu", "prior_sigma", "reward", "rpe", "precision", "value_estimate"):
        assert step[key] is not None, key
    valence, arousal = step["valence_arousal"]
    assert -1.0 <= valence <= 1.0
    assert -1.0 <= arousal <= 1.0


async def test_repeated_stimulus_sampling_is_bounded() -> None:
    # 不固定 seed、各自独立线程，重复同一刺激：e* 围绕后验抖动且方差有界、不发散。
    valences: list[float] = []
    for i in range(30):
        traj = await run(
            [Stimulus(name="x", goal_congruence=0.5, intensity=0.7)],
            thread_id=f"s{i}",
        )
        v, a = traj[0]["valence_arousal"]
        assert -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0
        valences.append(v)
    assert statistics.pstdev(valences) <= MAX_SAMPLE_SIGMA
    # 存在抖动（采样确有随机性），而非常数
    assert statistics.pstdev(valences) > 0.0


async def test_trace_accumulates_one_entry_per_node() -> None:
    # trace reducer（Annotated[list, operator.add]）应逐节点累加，而非被覆盖。
    graph = build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())
    result = await graph.ainvoke(
        {"stimulus": Stimulus(name="t", goal_congruence=0.5), "task_complete": False},
        config={"configurable": {"thread_id": "trace1"}},
    )
    nodes = [entry["node"] for entry in result["trace"]]
    # 默认 regulation 关闭：perception→appraisal→value→affect_core→expression→supervisor
    assert nodes == ["perception", "appraisal", "value", "affect_core", "expression", "supervisor"]
