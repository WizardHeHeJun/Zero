"""efference copy 副本落地测试（行为反馈环第一步）。

交接与验收标准：notes/2026-08-07-behavior-feedback-handoff.md §五。

核心断言：
1. 零回归："synth"/"directive" 不写 `motion_efference`；"efference" 下 `motion_directive`
   与 "directive" 模式逐字一致（副本纯增量）。
2. 副本一致性：`motion_efference == _efference_from_directive(directive)`；配变异验证——
   把副本任一字段改坏，一致性判据必须能红（绿灯先证明能红）。
3. 红线（副本不进情感内核）：人为塞入副本对 post_mu/affect_sample/mood 零影响；
   源码级守卫——内核数学模块不得引用 `motion_efference` 字段名（防将来有人未过议会接线）。
4. `deliberate_intents` 每轮归零（LastValue 残留防护，external_priors 先例）。
5. 副本跨回合持久（mood 模式）：不被每轮归零，MotionAgent 不写的回合保留上一回合值。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import src.agents.affect_core
import src.agents.affect_math
import src.agents.appraisal
import src.agents.mood
import src.agents.perception
import src.agents.value
from src.agents.motion import MotionAgent, _efference_from_directive
from src.orchestration.runner import ConversationSession
from src.orchestration.state import AffectState, Stimulus

agent = MotionAgent()


def _state(**kwargs: Any) -> AffectState:
    defaults: dict[str, Any] = {
        "motion_backend": "efference",
        "affect_sample": (0.3, -0.2),
        "regulated_affect": (0.1, 0.0),
        "language_text": "没错，就是这样。",
    }
    defaults.update(kwargs)
    return AffectState(**defaults)


# ── 1. 零回归 ────────────────────────────────────────────────────────────────


def test_synth_mode_never_writes_copy() -> None:
    assert agent(_state(motion_backend="synth")) == {}


def test_directive_mode_never_writes_copy() -> None:
    out = agent(_state(motion_backend="directive"))
    assert "motion_efference" not in out
    assert set(out.keys()) == {"motion_directive", "trace"}


def test_efference_directive_identical_to_directive_mode() -> None:
    """副本是纯增量：'efference' 下的 motion_directive 与 'directive' 模式逐字一致。"""
    out_directive = agent(_state(motion_backend="directive"))
    out_efference = agent(_state(motion_backend="efference"))
    assert out_efference["motion_directive"] == out_directive["motion_directive"]
    assert set(out_efference.keys()) == {"motion_directive", "trace", "motion_efference"}


# ── 2. 副本一致性 + 变异验证 ─────────────────────────────────────────────────


def test_copy_matches_directive_projection() -> None:
    out = agent(_state())
    assert out["motion_efference"] == _efference_from_directive(out["motion_directive"])


def test_copy_projection_shape() -> None:
    """口径定死：指令级、双通路系数 + scene + 离散事件（含 source 归因）。"""
    out = agent(_state())
    directive = out["motion_directive"]
    copy = out["motion_efference"]
    assert copy["spontaneous"] == {
        "amplitude": directive["amplitude"],
        "speed": directive["speed"],
        "onset": directive["onset"],
    }
    assert copy["voluntary"] == directive["regulated"]  # 非 None（regulated≠affect_sample）
    assert copy["voluntary"] is not directive["regulated"]  # 独立拷贝，不共享可变结构
    assert copy["scene"] == directive["scene"]
    assert [e["name"] for e in copy["events"]] == [e["name"] for e in directive["events"]]
    # direction 必须保真（head_tilt/glance 才有值；丢弃即第二步方向反馈返工，WARN-1）
    assert all(
        set(e.keys()) == {"name", "intensity", "direction", "source"} for e in copy["events"]
    )
    assert [e["direction"] for e in copy["events"]] == [e["direction"] for e in directive["events"]]


def _corrupt_amplitude(copy: dict[str, Any]) -> dict[str, Any]:
    copy["spontaneous"]["amplitude"] += 0.1
    return copy


def _corrupt_scene(copy: dict[str, Any]) -> dict[str, Any]:
    copy["scene"] = "speaking"
    return copy


def _corrupt_drop_events(copy: dict[str, Any]) -> dict[str, Any]:
    copy["events"] = []
    return copy


def _corrupt_voluntary_none(copy: dict[str, Any]) -> dict[str, Any]:
    copy["voluntary"] = None
    return copy


def _corrupt_event_intensity(copy: dict[str, Any]) -> dict[str, Any]:
    copy["events"][0]["intensity"] = 0.99
    return copy


def _corrupt_event_direction(copy: dict[str, Any]) -> dict[str, Any]:
    copy["events"][0]["direction"] = "right"
    return copy


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_amplitude,
        _corrupt_scene,
        _corrupt_drop_events,
        _corrupt_voluntary_none,
        _corrupt_event_intensity,
        _corrupt_event_direction,
    ],
)
def test_mutation_corrupted_copy_fails_consistency(corrupt: Any) -> None:
    """变异验证：副本任一字段被改坏，一致性判据（test_copy_matches_directive_projection
    的同一等式）必须能红——「在正确实现上会红」的断言才可信。"""
    out = agent(_state())
    good = out["motion_efference"]
    deep = {
        **good,
        "spontaneous": dict(good["spontaneous"]),
        "voluntary": dict(good["voluntary"]) if good["voluntary"] else None,
        "events": [dict(e) for e in good["events"]],
    }
    bad = corrupt(deep)
    assert bad != _efference_from_directive(out["motion_directive"])


# ── 3. 红线：副本不进情感内核 ─────────────────────────────────────────────────

_FAKE_COPY: dict[str, Any] = {
    "spontaneous": {"amplitude": 9.9, "speed": 9.9, "onset": 9.9},
    "voluntary": None,
    "scene": "idle",
    "events": [{"name": "nod", "intensity": 1.0, "source": "deliberate"}],
}


async def test_copy_injection_zero_influence_on_core() -> None:
    """同 seed 两会话，一边每轮人为塞入（极端值的）副本——内核输出必须逐字相等。
    若有人把 motion_efference 接进任何内核数学，本测试即红。"""
    stims = [
        Stimulus(name=f"s{i}", goal_congruence=g, intensity=0.8)
        for i, g in enumerate((0.6, -0.4, 0.2))
    ]
    plain = ConversationSession(thread_id="eff-influence-a", mood_enabled=True, rng_seed=11)
    injected = ConversationSession(thread_id="eff-influence-b", mood_enabled=True, rng_seed=11)
    for stim in stims:
        entry_plain = await plain.step(stim)
        entry_injected = await injected.step(stim, state_overrides={"motion_efference": _FAKE_COPY})
        assert entry_injected["valence_arousal"] == entry_plain["valence_arousal"]
        assert entry_injected["mood"] == entry_plain["mood"]
        assert entry_injected["prior_mu"] == entry_plain["prior_mu"]
        assert entry_injected["expression"] == entry_plain["expression"]


def test_core_math_modules_do_not_reference_copy() -> None:
    """源码级守卫（第二步设计门 PASS 后收窄口径·2026-08-07）：副本进内核**只允许**经
    议会裁定的受控入口——`affect_math.behavior_feedback_evidence`（+ `cap_stream_weight`
    后置封顶）与 `affect_core` workspace 分支的流装配调用点。其余内核模块
    （appraisal/value/mood/perception）**仍不得**引用 motion_efference：任何绕过受控入口
    的接线都未经议会评审（稳定性证明只覆盖受控路径）。
    第一步时期本守卫禁全部六模块（当时第二步未过议会）；本次收窄有据：
    notes/2026-08-07-behavior-feedback-council.md。"""
    for module in (
        src.agents.appraisal,
        src.agents.value,
        src.agents.mood,
        src.agents.perception,
    ):
        source = inspect.getsource(module)
        assert "motion_efference" not in source, f"{module.__name__} 引用了副本字段（红线）"


# ── 4/5. 会话级：快照、每轮归零、跨回合持久 ──────────────────────────────────


async def test_session_efference_snapshot_consistent() -> None:
    session = ConversationSession(thread_id="eff-snap", motion_backend="efference", rng_seed=7)
    await session.step(Stimulus(name="s1", goal_congruence=0.6, intensity=0.8))
    assert session.last_motion_directive is not None
    assert session.last_motion_efference is not None
    assert session.last_motion_efference == _efference_from_directive(session.last_motion_directive)


async def test_session_directive_mode_snapshot_stays_none() -> None:
    session = ConversationSession(thread_id="eff-dir", motion_backend="directive", rng_seed=7)
    await session.step(Stimulus(name="s1", goal_congruence=0.6, intensity=0.8))
    assert session.last_motion_directive is not None
    assert session.last_motion_efference is None


async def test_deliberate_intents_zeroed_each_turn() -> None:
    """每轮归零：轮 1 注入的 deliberate 意图不得残留到轮 2（LastValue 防护）。"""
    session = ConversationSession(thread_id="eff-zero", motion_backend="efference", rng_seed=7)
    await session.step(
        Stimulus(name="s1", goal_congruence=0.6, intensity=0.8),
        state_overrides={"deliberate_intents": [{"name": "nod", "intensity": 0.8}]},
    )
    assert session.last_motion_directive is not None
    assert any(e["source"] == "deliberate" for e in session.last_motion_directive["events"])
    await session.step(Stimulus(name="s2", goal_congruence=0.1, intensity=0.5))
    assert session.last_motion_directive is not None
    assert not any(e["source"] == "deliberate" for e in session.last_motion_directive["events"])


async def test_backend_switch_via_overrides_fails_fast() -> None:
    """staleness 次修正（议会 CS 席 2026-08-07）：motion_backend 是会话级固定门控，
    state_overrides 逐轮切换会破坏副本「恰好上一回合」语义——护栏 fail-fast 指向调用方。
    （本测试前身 test_copy_persists_across_turn_without_writer 验证的「陈旧值被静默保留」
    正是被本护栏封掉的行为。）"""
    session = ConversationSession(thread_id="eff-guard", motion_backend="efference", rng_seed=7)
    await session.step(Stimulus(name="s1", goal_congruence=0.6, intensity=0.8))
    with pytest.raises(ValueError, match="motion_backend"):
        await session.step(
            Stimulus(name="s2", goal_congruence=-0.3, intensity=0.6),
            state_overrides={"motion_backend": "synth"},
        )
    # 与会话 config 相同的显式传值放行（幂等无害）
    await session.step(
        Stimulus(name="s3", goal_congruence=0.1, intensity=0.5),
        state_overrides={"motion_backend": "efference"},
    )


def test_efference_without_affect_sample_clears_copy() -> None:
    """staleness 主修正：efference 档下前置缺失（affect_sample=None）不再裸 {}，
    显式返回 motion_efference=None——保证字段恒为「恰好上一回合」产出或 None，
    absent-cue 判定（is not None）无陈旧漏洞。directive 档保持旧行为（裸 {}）。"""
    state_eff = AffectState(motion_backend="efference", affect_sample=None)
    assert agent(state_eff) == {"motion_efference": None}
    state_dir = AffectState(motion_backend="directive", affect_sample=None)
    assert agent(state_dir) == {}
