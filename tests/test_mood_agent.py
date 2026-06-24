"""MoodAgent（A.7 心境更新节点）单测：门控、契约、与 mood_step 一致。"""

from __future__ import annotations

from src.agents.affect_math import mood_step
from src.agents.mood import MoodAgent
from src.orchestration.state import AffectState

FIELDS = set(AffectState.model_fields)


def test_mood_agent_noop_when_disabled() -> None:
    out = MoodAgent()(AffectState(affect_sample=(-0.5, 0.3), mood_enabled=False))
    assert out == {}


def test_mood_agent_noop_without_sample() -> None:
    assert MoodAgent()(AffectState(mood_enabled=True)) == {}


def test_mood_agent_updates_mood_and_matches_mood_step() -> None:
    state = AffectState(affect_sample=(-0.6, 0.4), mood=(-0.2, 0.1), mood_enabled=True)
    before = state.model_copy(deep=True)
    out = MoodAgent()(state)
    assert out["mood"] == mood_step((-0.2, 0.1), (-0.6, 0.4))
    assert set(out).issubset(FIELDS)  # 只返回合法 state 字段
    assert state == before  # 未原地 mutate 入参


def test_mood_agent_starts_from_neutral_when_no_prior_mood() -> None:
    out = MoodAgent()(AffectState(affect_sample=(-0.7, 0.2), mood_enabled=True))
    assert out["mood"] == mood_step((0.0, 0.0), (-0.7, 0.2))
