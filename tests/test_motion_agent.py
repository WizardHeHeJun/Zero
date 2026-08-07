"""MotionAgent 单测：门控/节点契约/scene 判定/events 抽取 + 图内零回归。

对应 `PRP/motion/design-agent.md`（已过议会门 2026-08-06 第五轮）。核心断言：
1. 门控默认 "synth" -> MotionAgent no-op、`ConversationSession.step()` 返回体
   （`_state_to_entry`）逐字不变 -- 零回归。
2. "directive" 开启后节点契约成立：只返回增量、缺前置字段返回 {}、产出可核验小结构。
3. scene 恒 "idle"（TTS 阻塞是刻意的，见 `MotionAgent._determine_scene`），
   即便本回合确有待发文本也不升级为 speaking。
4. events 走既有 12 词闭集（`behavior_intent`），词法与舞台说明两路均可命中。
"""

from __future__ import annotations

import pytest

from src.agents.motion import MotionAgent
from src.agents.motion_synth import modulation_from_affect
from src.memory.client import MemoryClient
from src.orchestration.graph import build_graph
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES, ConversationSession
from src.orchestration.state import AffectState, Stimulus
from src.storage.checkpointer import build_checkpointer

agent = MotionAgent()


def test_default_backend_synth_is_noop() -> None:
    state = AffectState(affect_sample=(0.4, 0.5))
    assert agent(state) == {}


def test_directive_without_affect_sample_returns_empty() -> None:
    state = AffectState(motion_backend="directive", affect_sample=None)
    assert agent(state) == {}


def test_directive_only_returns_increment_keys() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.2, 0.3))
    out = agent(state)
    assert set(out.keys()) == {"motion_directive", "trace"}


def test_directive_does_not_mutate_input_state() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.2, 0.3))
    before = state.model_copy(deep=True)
    agent(state)
    assert state == before


def test_directive_modulation_matches_analytic_fallback() -> None:
    affect = (0.3, -0.2)
    state = AffectState(motion_backend="directive", affect_sample=affect)
    out = agent(state)
    mod = modulation_from_affect(*affect)
    directive = out["motion_directive"]
    assert directive["amplitude"] == mod.amplitude
    assert directive["speed"] == mod.speed
    assert directive["onset"] == mod.onset_sharpness


def test_directive_regulated_none_without_regulated_affect() -> None:
    """未开调节（regulated_affect=None）→ directive["regulated"] 恒 None（结构最小）。"""
    state = AffectState(motion_backend="directive", affect_sample=(0.3, -0.2))
    directive = agent(state)["motion_directive"]
    assert directive["regulated"] is None


def test_directive_regulated_none_when_equal_to_affect() -> None:
    """regulated_affect 与 affect_sample 相等（调节没改变什么）→ 同样给 None。"""
    state = AffectState(
        motion_backend="directive", affect_sample=(0.3, -0.2), regulated_affect=(0.3, -0.2)
    )
    directive = agent(state)["motion_directive"]
    assert directive["regulated"] is None


def test_directive_regulated_matches_regulated_affect_modulation() -> None:
    """2026-08-07 修复：regulated_affect 与 affect_sample 不同时，directive["regulated"]
    须是**由 regulated_affect 单独算出**的一组系数，不是 affect_sample 那组的复制品。
    """
    affect = (-0.6, 0.8)
    regulated = (-0.2, 0.1)
    state = AffectState(
        motion_backend="directive", affect_sample=affect, regulated_affect=regulated
    )
    directive = agent(state)["motion_directive"]
    expected = modulation_from_affect(*regulated)
    assert directive["regulated"] == {
        "amplitude": expected.amplitude,
        "speed": expected.speed,
        "onset": expected.onset_sharpness,
    }
    # 不等于 spontaneous 那组（否则又是旧缺陷：两路共用一份）
    assert directive["regulated"] != {
        "amplitude": directive["amplitude"],
        "speed": directive["speed"],
        "onset": directive["onset"],
    }


def test_directive_is_deterministic() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.1, 0.6))
    out1 = agent(state)
    out2 = agent(state)
    assert out1 == out2


def test_directive_prosody_ref_always_none() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.0, 0.0))
    directive = agent(state)["motion_directive"]
    assert directive["prosody_ref"] is None


def test_scene_idle_without_pending_text() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.0, 0.0), language_text=None)
    directive = agent(state)["motion_directive"]
    assert directive["scene"] == "idle"


def test_scene_stays_idle_even_with_pending_text() -> None:
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.5, 0.7),
        language_text="jintian tianqi zhen hao, women chuqu zoulou ba.",
    )
    directive = agent(state)["motion_directive"]
    assert directive["scene"] == "idle"


def test_events_empty_without_language_text() -> None:
    state = AffectState(motion_backend="directive", affect_sample=(0.0, 0.0), language_text=None)
    assert agent(state)["motion_directive"]["events"] == []


def test_events_lexical_affirmation_triggers_nod() -> None:
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text="没错，就是这样。",
    )
    events = agent(state)["motion_directive"]["events"]
    names = {e["name"] for e in events}
    assert "nod" in names
    nod = next(e for e in events if e["name"] == "nod")
    assert nod["source"] == "lexical"


def test_events_stage_direction_routes_to_behavior() -> None:
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text="（点了点头）好的，我明白了。",
    )
    events = agent(state)["motion_directive"]["events"]
    names = {e["name"] for e in events}
    assert "nod" in names
    nod = next(e for e in events if e["name"] == "nod")
    assert nod["source"] == "stage"


def test_events_physical_world_claim_dropped() -> None:
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text="（我帮你把灯关了）好嘞。",
    )
    events = agent(state)["motion_directive"]["events"]
    assert events == []


# ── ③c deliberate 非文本意图源（行为反馈环第一步·缺口 B）──────────────────────


def test_events_deliberate_without_language_text() -> None:
    """deliberate 路不依赖文本——文本为空时上游意图照常下达（「先决定做个动作」）。"""
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text=None,
        deliberate_intents=[{"name": "nod", "intensity": 0.7}],
    )
    events = agent(state)["motion_directive"]["events"]
    assert len(events) == 1
    assert events[0]["name"] == "nod"
    assert events[0]["intensity"] == 0.7
    assert events[0]["source"] == "deliberate"


def test_events_deliberate_ranks_before_stage_and_lexical() -> None:
    """合并优先级 deliberate > stage > lexical：显式指令排最前。"""
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text="（挑了挑眉）没错，就是这样。",  # stage: brow_raise；lexical: nod
        deliberate_intents=[{"name": "lean_in"}],
    )
    events = agent(state)["motion_directive"]["events"]
    assert events[0]["name"] == "lean_in"
    assert events[0]["source"] == "deliberate"
    assert {e["name"] for e in events} >= {"lean_in", "brow_raise"}


def test_events_deliberate_same_name_dedups_over_text_routes() -> None:
    """同名去重时 deliberate 版本胜出（先入 seen）。"""
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        language_text="没错，就是这样。",  # lexical 也会出 nod（intensity 0.5）
        deliberate_intents=[{"name": "nod", "intensity": 0.9}],
    )
    events = agent(state)["motion_directive"]["events"]
    nods = [e for e in events if e["name"] == "nod"]
    assert len(nods) == 1
    assert nods[0]["source"] == "deliberate"
    assert nods[0]["intensity"] == 0.9


def test_deliberate_open_set_name_fails_fast() -> None:
    """闭集是安全边界：非 12 词（物理世界宣称）fail-fast，不静默丢弃——调用方是代码，
    静默会藏 bug（与 ③b 对模型产出的静默丢弃刻意不同）。"""
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        deliberate_intents=[{"name": "turn_off_light"}],
    )
    with pytest.raises(ValueError, match="闭集"):
        agent(state)


def test_deliberate_intensity_out_of_range_fails_fast() -> None:
    state = AffectState(
        motion_backend="directive",
        affect_sample=(0.2, 0.1),
        deliberate_intents=[{"name": "nod", "intensity": 1.5}],
    )
    with pytest.raises(ValueError, match="intensity"):
        agent(state)


def _make_graph():
    return build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())


async def test_graph_default_backend_motion_directive_none() -> None:
    graph = _make_graph()
    result = await graph.ainvoke(
        {"stimulus": Stimulus(name="x", goal_congruence=0.5, intensity=0.7)},
        config={"configurable": {"thread_id": "motion-default"}},
    )
    state = AffectState(**result)
    assert state.motion_directive is None
    assert all(entry.get("node") != "motion" for entry in state.trace)


async def test_graph_directive_backend_produces_directive() -> None:
    graph = _make_graph()
    result = await graph.ainvoke(
        {
            "stimulus": Stimulus(name="x", goal_congruence=0.5, intensity=0.7),
            "motion_backend": "directive",
        },
        config={"configurable": {"thread_id": "motion-directive"}},
    )
    state = AffectState(**result)
    assert state.motion_directive is not None
    assert state.motion_directive["scene"] == "idle"
    assert any(entry.get("node") == "motion" for entry in state.trace)


async def test_conversation_session_step_zero_regression_when_backend_default() -> None:
    session = ConversationSession(thread_id="motion-zero-regression")
    entry = await session.step(Stimulus(name="x", goal_congruence=0.5, intensity=0.7))
    assert "motion_directive" not in entry
    assert session.last_motion_directive is None


async def test_conversation_session_step_captures_last_motion_directive() -> None:
    session = ConversationSession(thread_id="motion-last-directive", motion_backend="directive")
    await session.step(Stimulus(name="x", goal_congruence=0.5, intensity=0.7))
    assert session.last_motion_directive is not None
    assert session.last_motion_directive["scene"] == "idle"


def test_session_config_rejects_invalid_motion_backend() -> None:
    from pydantic import ValidationError

    from src.orchestration.runner import SessionConfig

    with pytest.raises(ValidationError):
        SessionConfig(motion_backend="bogus")  # type: ignore[arg-type]


def test_session_config_default_motion_backend_is_synth() -> None:
    from src.orchestration.runner import SessionConfig

    assert SessionConfig().motion_backend == "synth"
