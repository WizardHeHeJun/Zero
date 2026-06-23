"""T6.1 节点契约：6 节点均 (state)->dict、只返回增量、不 mutate 入参。（G2）"""

from __future__ import annotations

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.appraisal import AppraisalAgent
from src.agents.expression import ExpressionAgent
from src.agents.perception import PerceptionAgent
from src.agents.regulation import RegulationAgent
from src.agents.value import ValueAgent
from src.memory.client import MemoryClient
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import SupervisorAgent

FIELDS = set(AffectState.model_fields)

SYNC_AGENTS = [
    PerceptionAgent(),
    AppraisalAgent(),
    ValueAgent(),
    AffectCoreAgent(),
    RegulationAgent(),
    ExpressionAgent(),
]


def full_state() -> AffectState:
    """各节点前置条件齐备的状态，确保每个节点都产出非空增量。"""
    return AffectState(
        stimulus=Stimulus(name="s", goal_congruence=0.6, intensity=0.8),
        prior_mu=(0.3, 0.4),
        prior_sigma=(0.1, 0.1),
        reward=0.6,
        rpe=0.5,
        precision=0.7,
        value_estimate=0.2,
        affect_sample=(0.3, 0.4),
        regulated_affect=(0.1, 0.2),
        rng_seed=1,
    )


@pytest.mark.parametrize("agent", SYNC_AGENTS)
def test_sync_node_returns_increment_dict(agent: object) -> None:
    state = full_state()
    before = state.model_copy(deep=True)
    out = agent(state)  # type: ignore[operator]
    assert isinstance(out, dict)
    assert set(out).issubset(FIELDS)  # 只返回合法 state 字段
    assert out, "节点应返回非空增量"
    assert state == before  # 未原地 mutate 入参


async def test_supervisor_node_contract() -> None:
    state = full_state()
    before = state.model_copy(deep=True)
    out = await SupervisorAgent(MemoryClient())(state)
    assert isinstance(out, dict)
    assert set(out).issubset(FIELDS)
    assert out.get("task_complete") is True
    assert state == before
