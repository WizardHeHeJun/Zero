"""T6.5 在线学习：同一刺激反复→V(s) 趋稳；断言 V(s)/后验未进 graph_store。（G4）"""

from __future__ import annotations

from src.agents.value import ValueAgent
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.runner import run
from src.orchestration.state import AffectState, Stimulus


def test_value_converges_on_repeated_reward() -> None:
    agent = ValueAgent()
    state = AffectState(stimulus=Stimulus(name="x"), reward=1.0)
    deltas: list[float] = []
    for _ in range(20):
        out = agent(state)
        deltas.append(out["rpe"])
        state = state.model_copy(update={"value_table": out["value_table"]})
    # RPE 幅度随经验下降并趋稳（在线 TD 收敛）
    assert abs(deltas[-1]) < abs(deltas[0])
    assert abs(deltas[-1]) < 0.05


async def test_runtime_state_not_leaked_into_graph_store() -> None:
    mem = MemoryClient()
    await run(
        [Stimulus(name="win", goal_congruence=0.8, intensity=0.9)],
        thread_id="g1",
        memory=mem,
        rng_seed=1,
    )
    # 经公开 query API 读取（不白盒访问后端内部），验证图谱只存事件/倾向摘要。
    # session_id 默认绑定 thread_id（"g1"），user_id 默认 "default-user"。
    session_facts = await mem.query("", scope=Scope.SESSION, key="g1")
    user_facts = await mem.query("", scope=Scope.USER, key="default-user")
    contents = [f.content for f in [*session_facts, *user_facts]]
    assert contents, "任务完成应写入至少一条记忆"
    # 运行态结构（value_table/后验）不得出现在长期记忆中
    assert all("value_table" not in c for c in contents)
    assert all("post_mu" not in c for c in contents)
    assert all("post_sigma" not in c for c in contents)
