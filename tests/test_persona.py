"""Persona「指定人格」接口单测：加载 + L1 人设卡 + L2 气质底色 + L3 预置关系。

三层全程默认中性 → 零回归（强断言：无 persona 时 converse system prompt 与改前逐字相等）。
全部 fake/内存，不打真 LLM / 网络 / torch。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from src.agents.affect_math import ATTITUDE_SETPOINT, EMOTION_REACTIVITY, EMOTION_RECOVERY
from src.agents.persona import Persona, load_persona
from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.chat_driver import ChatDriver
from src.storage.conversation_log import ConversationLog

# ---------------------------------------------------------------------------
# 加载 load_persona
# ---------------------------------------------------------------------------


def test_load_persona_neutral_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置任何项 → 中性 Persona()：setpoint/反应/恢复=引擎常量、无卡/无种子（零回归基线）。"""
    monkeypatch.delenv("ZERO_PERSONA_FILE", raising=False)
    p = load_persona()
    assert p == Persona()
    assert p.card == ""
    assert p.setpoint == ATTITUDE_SETPOINT
    assert p.reactivity == EMOTION_REACTIVITY
    assert p.recovery == EMOTION_RECOVERY
    assert p.initial_attitude is None
    assert p.seed_memories == ()


def test_load_persona_from_json_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """完整 JSON 文件：全字段解析（pair→tuple、seeds→tuple）。"""
    path = tmp_path / "persona.json"
    path.write_text(
        json.dumps(
            {
                "name": "小津",
                "card": "你叫小津，是老友",
                "setpoint": [0.1, -0.05],
                "reactivity": 0.7,
                "recovery": 0.5,
                "initial_attitude": [0.3, 0.1],
                "seed_memories": ["我们去过海边", "你不吃香菜"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(path))
    p = load_persona()
    assert p.name == "小津"
    assert p.card == "你叫小津，是老友"
    assert p.setpoint == (0.1, -0.05)
    assert p.reactivity == 0.7
    assert p.recovery == 0.5
    assert p.initial_attitude == (0.3, 0.1)
    assert p.seed_memories == ("我们去过海边", "你不吃香菜")


def test_load_persona_partial_file_keeps_neutral_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """部分字段文件：未给的字段退化为中性默认（缺 setpoint → 引擎常量）。"""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"card": "只配卡"}), encoding="utf-8")
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(path))
    p = load_persona()
    assert p.card == "只配卡"
    assert p.setpoint == ATTITUDE_SETPOINT
    assert p.initial_attitude is None


def test_load_persona_malformed_pair_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """显式给了文件却字段格式错（setpoint 非二元组）→ 抛错，不静默退化中性（fail-fast）。"""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"setpoint": [0.1]}), encoding="utf-8")
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(path))
    with pytest.raises(ValueError):
        load_persona()


def test_repo_persona_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """根目录 persona.example.json 模板：经 ZERO_PERSONA_FILE 可读出、card 非空、表「不编造」。"""
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "persona.example.json"
    assert example.exists()  # committed 模板（data/ 被 gitignore，故置于根目录）
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(example))
    p = load_persona()
    assert p.card  # L1 卡非空
    assert p.name  # 有名字
    assert "初次" in p.card or "不编造" in p.card  # 锁定「诚实陌生人」意图


# ---------------------------------------------------------------------------
# L1：人设卡注入对话 system prompt
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _CapturingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> _Resp:
        self.calls.append(dict(kwargs))
        return _Resp("回应")


class _CapturingChat:
    def __init__(self) -> None:
        self.completions = _CapturingCompletions()


class _CapturingClient:
    def __init__(self) -> None:
        self.chat = _CapturingChat()


async def test_l1_persona_prepended_to_system_prompt() -> None:
    """persona 非空 → converse 的 system prompt 以人设卡 + 空行起头（身份置于情绪行为框架前）。"""
    from src.agents.language_openai import OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x", persona="你叫小津")
    await lm.converse([{"role": "user", "content": "hi"}], (0.0, 0.0))
    system_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert system_content.startswith("你叫小津\n\n")


async def test_l1_no_persona_system_prompt_byte_identical() -> None:
    """persona 空（默认）→ system prompt 与改前逐字相等（零回归强断言）。"""
    from src.agents.emotion_lexicon import affect_label
    from src.agents.language_openai import _CONVERSE_SYS, OpenAILanguageModel

    client = _CapturingClient()
    lm = OpenAILanguageModel(client=client, model="x")  # persona 默认 ""
    await lm.converse([{"role": "user", "content": "hi"}], (0.0, 0.0))
    system_content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert system_content == _CONVERSE_SYS.format(feeling=affect_label(0.0, 0.0))


# ---------------------------------------------------------------------------
# L2/L3：ChatDriver 接气质底色 + 预置关系
# ---------------------------------------------------------------------------


class _FakeSession:
    """假 ConversationSession：step 返回固定 e*，无召回。"""

    def __init__(self, ev: tuple[float, float]) -> None:
        self.ev = ev

    async def step(self, stim: Any) -> dict[str, Any]:
        return {"valence_arousal": self.ev, "recalled_context": []}


class _RecordingSemantic:
    """记录 add_episode 全部入参的内存语义后端（search 恒空）。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def add_episode(
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
    ) -> None:
        self.calls.append(
            {"scope": scope, "key": key, "content": content, "embed_text": embed_text}
        )

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list:
        return []


def _make_driver(
    *,
    log: ConversationLog,
    session: Any,
    persona: Persona | None = None,
    memory: MemoryClient | None = None,
    seed_key: str = "",
    first_contact: bool = False,
    attitude: tuple[float, float] = (0.0, 0.0),
) -> ChatDriver:
    return ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=session,
        history=[],
        attitude=attitude,
        mode="test",
        persona=persona if persona is not None else Persona(),
        memory=memory,
        seed_key=seed_key,
        first_contact=first_contact,
    )


async def test_l2_warm_setpoint_lifts_emotion_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中性刺激下，偏暖 setpoint 的 persona 情绪基线被抬高（气质底色真生效）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    neutral = _make_driver(log=log, session=_FakeSession((0.0, 0.0)))
    warm = _make_driver(
        log=log, session=_FakeSession((0.0, 0.0)), persona=Persona(setpoint=(0.5, 0.0))
    )
    tn = await neutral.step("x")
    tw = await warm.step("x")
    assert tw.emotion[0] > tn.emotion[0]
    log.close()


async def test_l2_default_persona_zero_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认中性 Persona() 下，正向 e* 两时间尺度都从 0 推进（与改前 test_chat_driver 行为一致）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    driver = _make_driver(log=log, session=_FakeSession((0.8, 0.3)))
    turn = await driver.step("好棒")
    assert turn.attitude != (0.0, 0.0)
    assert turn.emotion != (0.0, 0.0)
    log.close()


async def test_l3_seeds_memories_on_first_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次接触 + 种子 → 首轮按 USER 作用域 + seed_key 幂等写入；二轮不重播。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    sem = _RecordingSemantic()
    mem = MemoryClient(semantic=sem)
    persona = Persona(seed_memories=("我们去过海边", "你不吃香菜"))
    driver = _make_driver(
        log=log,
        session=_FakeSession((0.0, 0.0)),
        persona=persona,
        memory=mem,
        seed_key="alice",
        first_contact=True,
    )
    await driver.step("hi")
    assert len(sem.calls) == 2
    assert all(c["scope"] == Scope.USER.value for c in sem.calls)
    assert all(c["key"] == "alice" for c in sem.calls)
    assert sem.calls[0]["embed_text"] == "我们去过海边"  # 嵌入纯文本
    assert "precision=" in sem.calls[0]["content"]
    assert "first_contact=True" in sem.calls[0]["content"]
    await driver.step("再说一句")  # 二轮：seeded 守卫 → 不再写
    assert len(sem.calls) == 2
    log.close()


async def test_l3_no_seed_when_not_first_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非首次接触（已有 transcript）→ 即便有种子也不播种（不覆盖既有关系）。"""
    monkeypatch.setattr("src.orchestration.chat_driver.random.gauss", lambda *a: 0.0)
    log = ConversationLog(":memory:")
    sem = _RecordingSemantic()
    mem = MemoryClient(semantic=sem)
    driver = _make_driver(
        log=log,
        session=_FakeSession((0.0, 0.0)),
        persona=Persona(seed_memories=("旧关系记忆",)),
        memory=mem,
        seed_key="bob",
        first_contact=False,
    )
    await driver.step("hi")
    assert sem.calls == []
    log.close()
