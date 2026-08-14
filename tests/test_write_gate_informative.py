"""写入门第四通道 `is_informative`（PRP/write-gate-informative）单测。

覆盖 design.md 候选 a `_appraise` 重锚点版·8 条前置的落地面：

  T1. `OpenAILanguageModel._parse_vad_informative` 解析/降级（同 `_parse_vad` 同构断言）。
  T2. `appraise_text_informative` 端到端：同一次调用产 (v,a,informative)，零新增调用。
  T3. ChatDriver 鸭子探测：门关/门开/lm 无该方法三种路径，及 `state_overrides` 注入。
  T4. `ConversationSession.step` 跨轮归零（LastValue 残留防线，仿 text_coping_prior 先例）。
  T5. SupervisorAgent 写入门第四条：默认关零回归、门开命中/不命中、首因名额不重复消费、
      节流（全轮仅一次 write_episode）。
  T6. 只写不消费：`informative=True` 对 `parse_importance_tags` / `importance_signal` /
      `combine_importance_with_precision` 的读数逐字无影响（design.md §三·前置 4 的机械锁）。

⚠ 变异验证记录（手工做，未固化为自动化测试，见 docstring 底部）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai")

from src.agents.language_openai import OpenAILanguageModel  # noqa: E402
from src.memory.client import MemoryClient  # noqa: E402
from src.memory.types import Scope  # noqa: E402
from src.memory.utils import (  # noqa: E402
    combine_importance_with_precision,
    importance_signal,
    parse_importance_tags,
)
from src.orchestration.chat_driver import ChatDriver  # noqa: E402
from src.orchestration.state import AffectState, Stimulus  # noqa: E402
from src.orchestration.supervisor import SupervisorAgent  # noqa: E402
from src.storage.conversation_log import ConversationLog  # noqa: E402

ENV_KEY = "ZERO_WRITE_GATE_INFORMATIVE"

# ---------------------------------------------------------------------------
# T1. _parse_vad_informative 单测
# ---------------------------------------------------------------------------


class TestParseVadInformative:
    def test_normal_json_three_values(self) -> None:
        raw = '{"valence": 0.4, "arousal": -0.2, "informative": true}'
        assert OpenAILanguageModel._parse_vad_informative(raw) == (0.4, -0.2, True)

    def test_informative_false_roundtrips(self) -> None:
        raw = '{"valence": 0.1, "arousal": 0.1, "informative": false}'
        assert OpenAILanguageModel._parse_vad_informative(raw) == (0.1, 0.1, False)

    def test_informative_missing_defaults_false_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw = '{"valence": 0.4, "arousal": -0.2}'
        with caplog.at_level("WARNING"):
            result = OpenAILanguageModel._parse_vad_informative(raw)
        assert result == (0.4, -0.2, False)
        assert "informative" in caplog.text

    @pytest.mark.parametrize("informative_literal", ['"yes"', "1", "null", "0"])
    def test_informative_non_bool_defaults_false(self, informative_literal: str) -> None:
        raw = f'{{"valence": 0.15, "arousal": -0.15, "informative": {informative_literal}}}'
        v, a, informative = OpenAILanguageModel._parse_vad_informative(raw)
        assert (v, a) == (0.15, -0.15)
        assert informative is False

    def test_malformed_json_defaults_all(self) -> None:
        assert OpenAILanguageModel._parse_vad_informative("not json at all") == (0.0, 0.0, False)

    def test_no_json_braces_defaults_all(self) -> None:
        assert OpenAILanguageModel._parse_vad_informative("valence 0.5") == (0.0, 0.0, False)

    def test_va_degrade_path_matches_parse_vad(self) -> None:
        """v/a 降级路径与 `_parse_vad` 逐字同构（含越界钳制）——不重复实现的机械锁。"""
        raws = [
            '{"valence": 5.0, "arousal": -5.0, "informative": true}',  # 越界钳制
            "garbage",  # 无 JSON
            '{"valence": "nope"}',  # 类型错误
        ]
        for raw in raws:
            va_only = OpenAILanguageModel._parse_vad(raw)
            va_from_informative = OpenAILanguageModel._parse_vad_informative(raw)[:2]
            assert va_only == va_from_informative, raw


# ---------------------------------------------------------------------------
# T2. appraise_text_informative 端到端：零新增调用 + 同一次 parse
# ---------------------------------------------------------------------------


class _JsonClient:
    """假 OpenAI client：固定回一段原始 JSON 文本，记录调用次数与最近一次 kwargs。"""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.call_count = 0
        self.last_kwargs: dict[str, Any] = {}
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.call_count += 1
                outer.last_kwargs = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=outer.raw))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


async def test_appraise_text_informative_single_call_produces_triple() -> None:
    client = _JsonClient('{"valence": 0.5, "arousal": 0.2, "informative": true}')
    lm = OpenAILanguageModel(model="test-model", client=client)
    result = await lm.appraise_text_informative("我明天要交房租")
    assert result == (0.5, 0.2, True)
    assert client.call_count == 1, "同一次调用产三值——不得叠加 appraise_text 的第二次调用"
    assert client.last_kwargs["temperature"] == 0.0


async def test_appraise_text_informative_missing_field_warns_and_defaults_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _JsonClient('{"valence": -0.2, "arousal": 0.1}')
    lm = OpenAILanguageModel(model="test-model", client=client)
    with caplog.at_level("WARNING"):
        result = await lm.appraise_text_informative("随便聊聊")
    assert result == (-0.2, 0.1, False)
    assert client.call_count == 1


# ---------------------------------------------------------------------------
# T3. ChatDriver 鸭子探测 + state_overrides 注入
# ---------------------------------------------------------------------------


class _FakeLmWithInformative:
    """带 appraise_text_informative 的假 lm：记录两个方法各被调次数。"""

    def __init__(self, *, v: float = 0.0, a: float = 0.0, informative: bool = True) -> None:
        self.v = v
        self.a = a
        self.informative = informative
        self.appraise_text_calls = 0
        self.appraise_text_informative_calls = 0

    async def appraise_text(self, text: str) -> tuple[float, float]:
        self.appraise_text_calls += 1
        return (self.v, self.a)

    async def appraise_text_informative(self, text: str) -> tuple[float, float, bool]:
        self.appraise_text_informative_calls += 1
        return (self.v, self.a, self.informative)

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
        relationship_hint: str = "",
    ) -> str:
        return "ok"


class _FakeLmNoInformative:
    """仿 SteeringLanguageModel 形态的假 lm：只满足 ConversationModel，无第四通道方法。"""

    def __init__(self, *, v: float = 0.0, a: float = 0.0) -> None:
        self.v = v
        self.a = a
        self.appraise_text_calls = 0

    async def appraise_text(self, text: str) -> tuple[float, float]:
        self.appraise_text_calls += 1
        return (self.v, self.a)

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
        relationship_hint: str = "",
    ) -> str:
        return "ok"


class _FakeCapturingSession:
    """假 ConversationSession：记录每轮传入的 state_overrides（None 或 dict）。"""

    def __init__(self, ev: tuple[float, float] = (0.0, 0.0)) -> None:
        self.ev = ev
        self.captured_overrides: list[dict[str, Any] | None] = []

    async def step(
        self, stim: Any, state_overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.captured_overrides.append(state_overrides)
        return {"valence_arousal": self.ev, "recalled_context": []}


def _make_driver(*, lm: Any, session: Any, log: ConversationLog) -> ChatDriver:
    return ChatDriver(
        thread="t",
        lm=lm,
        log=log,
        session=session,
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
    )


async def test_default_off_chat_driver_calls_only_appraise_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3-a 默认关零回归：env 未设 → 恒走 appraise_text，不碰新方法（即便 lm 具备它）。"""
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    lm = _FakeLmWithInformative(informative=True)
    session = _FakeCapturingSession()
    driver = _make_driver(lm=lm, session=session, log=log)
    await driver.step("我明天下午两点有个会")
    assert lm.appraise_text_calls == 1
    assert lm.appraise_text_informative_calls == 0
    assert session.captured_overrides[-1] is None, (
        "门关时 is_informative_hint 恒不注入，state_overrides 应为空 dict → session.step(stim) "
        "单参调用（本 fake 的默认参数落 None）"
    )
    log.close()


async def test_gate_open_hint_true_injects_state_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3-b 门开 + 命中：走扩展方法，state_overrides 含 is_informative_hint=True。"""
    monkeypatch.setenv(ENV_KEY, "1")
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    lm = _FakeLmWithInformative(informative=True)
    session = _FakeCapturingSession()
    driver = _make_driver(lm=lm, session=session, log=log)
    await driver.step("我住在朝阳区")
    assert lm.appraise_text_informative_calls == 1
    assert lm.appraise_text_calls == 0, "门开时不得叠加原 appraise_text 调用（替代而非叠加）"
    assert session.captured_overrides[-1] == {"is_informative_hint": True}
    log.close()


async def test_gate_open_hint_false_does_not_inject(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3-c 门开 + 不命中：扩展方法仍被调，但 state_overrides 不注入该键（第四条不误触发）。"""
    monkeypatch.setenv(ENV_KEY, "1")
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    lm = _FakeLmWithInformative(informative=False)
    session = _FakeCapturingSession()
    driver = _make_driver(lm=lm, session=session, log=log)
    await driver.step("嗯嗯好的")
    assert lm.appraise_text_informative_calls == 1
    assert session.captured_overrides[-1] is None
    log.close()


async def test_gate_open_lm_without_new_method_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3-d 鸭子探测回退：门开但 lm 无 appraise_text_informative → 走 appraise_text，
    hint 恒 False（design.md 热路径边界：标记只能产自 LLM 已在场节点，非强行造调用）。
    """
    monkeypatch.setenv(ENV_KEY, "1")
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    lm = _FakeLmNoInformative()
    session = _FakeCapturingSession()
    driver = _make_driver(lm=lm, session=session, log=log)
    await driver.step("随便聊聊")
    assert lm.appraise_text_calls == 1
    assert not hasattr(lm, "appraise_text_informative")
    assert session.captured_overrides[-1] is None
    log.close()


# ---------------------------------------------------------------------------
# T4. ConversationSession.step 跨轮归零（LastValue 残留防线）
# ---------------------------------------------------------------------------


async def test_is_informative_hint_zeroed_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """仿 test_text_coping_prior.py::test_text_coping_prior_zeroed_each_step 的 spy 手法。

    第1轮：state_overrides 注入 True → base 里应为 True。
    第2轮：不注入 → base 必须显式归零为 False，否则从 checkpoint 继承第1轮残留。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig

    cfg = SessionConfig(rng_seed=42, affect_readout="map")
    stim = Stimulus(name="test_zero", goal_congruence=0.2, intensity=0.5)
    session = ConversationSession(thread_id="t-informative-zero", config=cfg)

    captured: list[dict[str, Any]] = []
    orig_ainvoke = session.graph.ainvoke

    async def _spy(base: dict, *args: object, **kwargs: object) -> object:
        captured.append(dict(base))
        return await orig_ainvoke(base, *args, **kwargs)

    monkeypatch.setattr(session.graph, "ainvoke", _spy)

    await session.step(stim, state_overrides={"is_informative_hint": True})
    assert captured[-1]["is_informative_hint"] is True

    await session.step(stim, state_overrides=None)
    assert captured[-1]["is_informative_hint"] is False, (
        "第2轮不传 state_overrides 时 is_informative_hint 必须显式归零为 False；"
        "非 False = LastValue checkpoint 残留"
    )


# ---------------------------------------------------------------------------
# T5. SupervisorAgent 写入门第四条
# ---------------------------------------------------------------------------


class _RecordingStore:
    """记录 add_episode 调用（含 importance_tag）的 fake SemanticStore（仿 identity_fact 先例）。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def add_episode(
        self,
        content: str,
        *,
        scope: str,
        key: str,
        valid_at: object = None,
        embed_text: str | None = None,
        importance_tag: float | None = None,
    ) -> None:
        self.calls.append(
            {"content": content, "scope": scope, "key": key, "importance_tag": importance_tag}
        )

    async def search(self, *args: object, **kwargs: object) -> list:
        return []


# 低 salience（precision × |rpe| 远低于门 0.15）、非承诺、非身份自陈的中性文本。
_BENIGN_TEXT = "今天天气不错，我出去转了一圈"


def _make_state(
    text: str = _BENIGN_TEXT,
    *,
    precision: float = 1.0,
    rpe: float = 0.0,
    user_id: str = "u1",
    is_informative_hint: bool = False,
) -> AffectState:
    """构造一轮 supervisor 输入。默认 salience = 1.0×0.0 = 0.0 < 门 0.15 ⇒ 主门不写，
    从而单独检验第四条。"""
    return AffectState(
        stimulus=Stimulus(name=text[:40], text=text, goal_congruence=0.0, intensity=0.0),
        affect_sample=(0.0, 0.0),
        affect_precision=precision,
        rpe=rpe,
        user_id=user_id,
        is_informative_hint=is_informative_hint,
    )


async def test_default_off_supervisor_ignores_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认关零回归：env 未设 → 即便 state.is_informative_hint=True 也不因第四条写入。"""
    monkeypatch.delenv(ENV_KEY, raising=False)
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    assert agent.informative_gate_enabled is False
    await agent(_make_state(is_informative_hint=True))
    assert store.calls == [], "门关时低 salience + hint=True 不应触发写入"


async def test_gate_open_hint_true_writes_with_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """门开 + 低 salience + hint=True → 写入且 content 含 ` | informative=True`。"""
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    assert agent.informative_gate_enabled is True
    await agent(_make_state(is_informative_hint=True))
    assert len(store.calls) == 1
    assert " | informative=True" in store.calls[0]["content"]


async def test_gate_open_hint_false_does_not_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """门开 + 低 salience + hint=False → 不写（第四条不误触发）。"""
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state(is_informative_hint=False))
    assert store.calls == []


async def test_gate_open_first_contact_slot_not_reconsumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首因名额不重复消费：同一 user 第二次经第四条触发写入，不再打 first_contact=True。"""
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state(is_informative_hint=True, user_id="u1"))
    await agent(_make_state("刚整理了一下桌子", is_informative_hint=True, user_id="u1"))
    assert len(store.calls) == 2
    assert " | first_contact=True" in store.calls[0]["content"]
    assert " | first_contact=True" not in store.calls[1]["content"], (
        "首因名额已被第一次写入消费，第二次不应重复打标"
    )


async def test_gate_open_overlapping_hits_write_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """节流：informative 与既有 salience 主门同时命中时，全轮仍只有一次 write_episode。"""
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    # precision×|rpe| = 10×0.9 = 9.0 远超门 0.15，主门自身即会写；同轮再叠加 hint=True。
    await agent(_make_state(precision=10.0, rpe=0.9, is_informative_hint=True))
    assert len(store.calls) == 1, f"重叠命中应仅写一次，实际 {len(store.calls)}"
    assert " | informative=True" in store.calls[0]["content"]


# ---------------------------------------------------------------------------
# T6. 只写不消费：informative 段对既有解析/组合函数逐字无影响
# ---------------------------------------------------------------------------


async def test_informative_tag_not_registered_in_parse_importance_tags() -> None:
    """`parse_importance_tags` 恒只含三键，不会因 informative=True 多出第四键。"""
    content = "gist | 情绪=中性(0.00,0.00) | precision=10.00 | precision_raw=10.00 | value=0.100"
    tags_without = parse_importance_tags(content)
    tags_with = parse_importance_tags(content + " | informative=True")
    assert set(tags_with) == {"first_contact", "commitment", "identity"}
    assert tags_without == tags_with


async def test_informative_tag_does_not_change_importance_signal() -> None:
    content = (
        "gist | 情绪=中性(0.00,0.00) | precision=10.00 | precision_raw=10.00 "
        "| value=0.100 | first_contact=True"
    )
    tags_without = parse_importance_tags(content)
    tags_with = parse_importance_tags(content + " | informative=True")
    assert importance_signal(tags_without) == importance_signal(tags_with)


async def test_informative_tag_does_not_change_combine_importance_with_precision() -> None:
    content = (
        "gist | 情绪=中性(0.00,0.00) | precision=10.00 | precision_raw=10.00 "
        "| value=0.100 | commitment=True | identity=name"
    )
    content_with_tag = content + " | informative=True"
    i_without = combine_importance_with_precision(
        importance_signal(parse_importance_tags(content)), content
    )
    i_with = combine_importance_with_precision(
        importance_signal(parse_importance_tags(content_with_tag)), content_with_tag
    )
    assert i_without == i_with


async def test_end_to_end_informative_segment_isolated_from_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端：真实 supervisor 写入的 episode content，informative 段对消费方读数无影响。

    两次写入 salience 主门都触发（precision×|rpe|=9.0），只有 hint 不同 → 两条 content
    仅在 informative 段上有差异；tag 解析结果、importance_signal、
    combine_importance_with_precision、write_episode 传给存储层的 importance_tag 逐字相同。
    """
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent_a = SupervisorAgent(MemoryClient(semantic=store))
    agent_b = SupervisorAgent(MemoryClient(semantic=store))
    await agent_a(_make_state(precision=10.0, rpe=0.9, is_informative_hint=False, user_id="userA"))
    await agent_b(_make_state(precision=10.0, rpe=0.9, is_informative_hint=True, user_id="userB"))

    content_without = store.calls[0]["content"]
    content_with = store.calls[1]["content"]
    assert " | informative=True" in content_with
    assert " | informative=True" not in content_without

    tags_without = parse_importance_tags(content_without)
    tags_with = parse_importance_tags(content_with)
    assert tags_without == tags_with

    i_without = combine_importance_with_precision(importance_signal(tags_without), content_without)
    i_with = combine_importance_with_precision(importance_signal(tags_with), content_with)
    assert i_without == i_with

    assert store.calls[0]["importance_tag"] == store.calls[1]["importance_tag"], (
        "write_episode 传给存储层的 importance_tag（纯 tag 分量）不应被 informative 段影响"
    )


async def test_identity_episode_still_uses_user_scope_with_gate_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """门开不改变既有 scope 契约：episode 仍写 Scope.USER（防第四条引入 scope 回归）。"""
    monkeypatch.setenv(ENV_KEY, "1")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state(is_informative_hint=True))
    assert store.calls[0]["scope"] == Scope.USER


# ---------------------------------------------------------------------------
# 变异验证记录（手工做，非固化自动化测试；见 PRP/write-gate-informative/design.md）
# ---------------------------------------------------------------------------
#
# ①去掉 supervisor.py 第四条 OR（把 `informative_hit` 从 `if salient or is_commitment or
#   identity_hit or informative_hit:` 里删掉）：test_gate_open_hint_true_writes_with_tag 与
#   test_gate_open_overlapping_hits_write_exactly_once 均转红（前者 store.calls==[]，
#   后者仍绿但内容不含 informative=True——用第一条即可判别）。手工验证通过，已还原。
#
# ②把 chat_driver.py 的门控判断去掉（`appraise_informative = getattr(self.lm,
#   "appraise_text_informative", None) if self.lm is not None else None`，即恒探测新方法
#   不再受 write_gate_informative_enabled 约束）：test_default_off_chat_driver_calls_only_
#   appraise_text 转红（appraise_text_informative_calls 从 0 变 1，appraise_text_calls
#   从 1 变 0）。手工验证通过，已还原。
#
# 两次变异均在验证后用 Edit 精确还原，`git diff -- src/` 干净。
