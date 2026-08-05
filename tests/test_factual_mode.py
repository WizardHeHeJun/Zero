"""事实化模式（`ZERO_FACTUAL_MODE`）：不捏造身份/环境/往事，但保住引擎驱动的情绪。

背景（2026-07-31 100 轮实跑）：数字人捏造**自己的**身份与处境——报出具体日期「20号，周二」、
描述身体动作「我刚走到窗边撩开帘子」、编出姓名职业「沈念／写东西的」，并把用户从未说过的话
当成往事引用。审计定位到四条独立成因，本文件逐条上锁：

  A. 身份断言：`_TEMPER_ADDENDUM` 的「你是一个……的人」+ `_CONVERSE_SYS_TAIL` 的「像真人」
     —— 模型据此用人类图式补全一切它没有的属性。
  B. 诚实条款作用域：只覆盖「历史里找不到」的*记忆缺口*，不覆盖「根本无从知道」的*认知边界*；
     同段「平常心答即可」反而是作答许可证。
  C. push 词表无中性死区：`suggest_affect_words` 只按内积排序，e* 模长趋零时方向纯属噪声——
     实测 e*=(-0.079,+0.037)（`affect_label` 判「平静」）取出「暴怒/愤怒/恐惧」。
  D. 召回残留：`runner` 每轮基准漏归零 `recalled_context`/`recalled_facts`，而
     `MemoryRecallAgent` 只在命中时才写 —— 未命中轮会读到**上一轮**的记忆。
     ⚠ `ConversationSession.step` 与 `run()` 是**两条平行入口**，归零基准必须同步
     （run() 共享 thread_id 跨 stimulus 持久化，同一残留可原样复现）；同款条件写入的
     `recalled_disposition` 直接偏置 appraisal 的 prior_mu，一并归零。

  成因 C 的死区判定下沉在 `suggest_affect_words` 内（env `ZERO_PUSH_NEUTRAL_DEADZONE`
  全局默认 + 调用点显式 bool 压过），三个调用点（converse/_compose/模板模型）统一受控，
  不再绑死在事实化模式上。

⚠ 默认关 = 逐字零回归。**删的是身份，不是情绪**：反塌陷断言（`test_open_gate_keeps_*`）
专门防后来者为了压捏造把情绪条款一起删掉。

快照夹具 `fixtures_default_converse_prompt.py` 由 scratchpad/gen_prompt_fixture.py 生成：
读三段默认常量拼接后按 42 字宽分块写出，附长度与 sha256。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai")

from src.agents.emotion_lexicon import (  # noqa: E402
    NEUTRAL_RADIUS,
    affect_label,
    suggest_affect_words,
)
from src.agents.language_openai import (  # noqa: E402
    _CONVERSE_SYS_HEAD,
    _CONVERSE_SYS_TAIL,
    _FACT_BOUNDARY_ADDENDUM,
    _FACTUAL_SYS_HEAD,
    _FACTUAL_SYS_TAIL,
    _FACTUAL_TEMPER_ADDENDUM,
    _PUSH_ADDENDUM,
    _TEMPER_ADDENDUM,
    OpenAILanguageModel,
    factual_mode_enabled,
)
from tests.fixtures_default_converse_prompt import (  # noqa: E402
    DEFAULT_CONVERSE_PROMPT_LEN,
    DEFAULT_CONVERSE_PROMPT_SHA256,
    DEFAULT_CONVERSE_SYSTEM_PROMPT,
)

HISTORY = [{"role": "user", "content": "外面还在下雨吗"}]
CALM = (0.09, -0.03)  # 实跑里真实出现过的「平静」轮 e*
NEGATIVE = (-0.30, 0.10)


class _CapturingClient:
    """捕获 chat.completions.create 收到的 messages，不发网络请求。"""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.captured = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


async def _system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factual: str | None,
    affect: tuple[float, float] = CALM,
    retrieved: str = "",
    push: bool = False,
    relationship_hint: str = "",
    persona: str = "",
) -> str:
    if factual is None:
        monkeypatch.delenv("ZERO_FACTUAL_MODE", raising=False)
    else:
        monkeypatch.setenv("ZERO_FACTUAL_MODE", factual)
    monkeypatch.delenv("ZERO_TEMPER_VALENCE_GATE", raising=False)
    client = _CapturingClient()
    lm = OpenAILanguageModel(model="test-model", client=client, persona=persona)
    await lm.converse(HISTORY, affect, retrieved, push=push, relationship_hint=relationship_hint)
    return client.captured["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 1. 开关解析：只有白名单值开门（防 `=false` 反被判成开）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " 1 ", "On"])
def test_truthy_values_open_the_gate(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("ZERO_FACTUAL_MODE", val)
    assert factual_mode_enabled() is True


@pytest.mark.parametrize("val", ["", "0", "false", "FALSE", "off", "no", "  ", "2", "enabled"])
def test_falsy_values_keep_the_gate_shut(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    """⚠ `not in ("", "0")` 式写法会把 `false`/`off`/`no` 判成开——这里逐个钉死。"""
    monkeypatch.setenv("ZERO_FACTUAL_MODE", val)
    assert factual_mode_enabled() is False, f"{val!r} 不该开门"


def test_unset_keeps_the_gate_shut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_FACTUAL_MODE", raising=False)
    assert factual_mode_enabled() is False


# ---------------------------------------------------------------------------
# 2. 零回归：默认关时 system prompt 与改前**逐字**相同
# ---------------------------------------------------------------------------


def test_default_prompt_matches_literal_snapshot() -> None:
    """默认三段拼接必须与字面量快照逐字相同。

    这条补的是 test_persona.py 那条断言的盲区：它拿常量自身拼接算 expected，
    改了常量文本它照样绿。本条钉死字面量，改一个字就红。
    """
    actual = _CONVERSE_SYS_HEAD + _TEMPER_ADDENDUM + _CONVERSE_SYS_TAIL
    assert len(actual) == DEFAULT_CONVERSE_PROMPT_LEN, (
        f"默认 prompt 长度从 {DEFAULT_CONVERSE_PROMPT_LEN} 变成 {len(actual)}；"
        "若确实要改默认行为，同批更新 fixtures_default_converse_prompt.py 并说明理由"
    )
    assert hashlib.sha256(actual.encode("utf-8")).hexdigest() == DEFAULT_CONVERSE_PROMPT_SHA256
    assert actual == DEFAULT_CONVERSE_SYSTEM_PROMPT


async def test_gate_shut_prompt_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """未开门 → system prompt == 默认快照 .format(feeling=…)，一字不差。"""
    sys_prompt = await _system_prompt(monkeypatch, factual=None)
    assert sys_prompt == DEFAULT_CONVERSE_SYSTEM_PROMPT.format(feeling=affect_label(*CALM))


async def test_gate_shut_recall_lead_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """未开门 → 召回仍用原「你还记得以下背景：」引子。"""
    sys_prompt = await _system_prompt(monkeypatch, factual=None, retrieved="片段A")
    assert "你还记得以下背景：片段A" in sys_prompt


async def test_gate_shut_push_addendum_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """未开门 → push 段仍是原文（含「不自觉地」），且平静轮**照旧**注入（无死区）。"""
    sys_prompt = await _system_prompt(monkeypatch, factual=None, push=True)
    assert "此刻你不自觉地，用词会偏向这类词的色彩" in sys_prompt


# ---------------------------------------------------------------------------
# 3. 开门：身份断言被摘掉、边界段在最末
# ---------------------------------------------------------------------------


async def test_open_gate_removes_person_identity_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    """开门 → 三处「你是人/像真人」的身份断言全部消失。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", affect=NEGATIVE)
    assert "像真人" not in sys_prompt
    assert "有情绪起伏**的人" not in sys_prompt
    assert "那是讨好型客服，不是真人" not in sys_prompt


async def test_open_gate_states_it_is_a_program(monkeypatch: pytest.MonkeyPatch) -> None:
    sys_prompt = await _system_prompt(monkeypatch, factual="1")
    assert "你是一个 AI 程序" in sys_prompt
    assert "没有身体、没有感官" in sys_prompt
    assert "这不是情景扮演" in sys_prompt


async def test_boundary_section_is_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """边界段必须**恰好一次**且拼在**最末**（最强近因位）——即便人设卡/召回/push 全都在场。

    endswith 单独锁不住「挪到 push 之前」：若挪动写成「前面加一处、末尾没删」，
    结尾仍是它、照样绿（变异实测）。补 count==1 后，重复注入与真移动都会驱红。
    """
    sys_prompt = await _system_prompt(
        monkeypatch,
        factual="1",
        affect=NEGATIVE,
        retrieved="片段A",
        push=True,
        relationship_hint="已经比较熟络",
        persona="你叫小津",
    )
    assert sys_prompt.count(_FACT_BOUNDARY_ADDENDUM) == 1, "边界段应恰好出现一次"
    assert sys_prompt.endswith(_FACT_BOUNDARY_ADDENDUM), "边界段不在最末位"


async def test_open_gate_covers_knowledge_boundary_not_just_memory_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成因 B：诚实条款必须区分「记忆缺口」与「认知边界」，并禁止用前者兜后者。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1")
    assert "我这边没有这个信息" in sys_prompt
    assert "别拿「我不记得了」蒙混" in sys_prompt
    assert "平常心答即可" not in sys_prompt, "「平常心答即可」是「20号，周二」的作答许可证"


@pytest.mark.parametrize(
    "probe",
    [
        "没有时钟也没有日历",
        "你没有传感器",
        "只能看到他打过来的文字",
        "你自己的姓名、年龄、职业",
        "只有这段上下文里",
    ],
)
async def test_open_gate_enumerates_all_five_unknowable_classes(
    monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    """五类不可知信息逐条在场（日期/天气/对方环境/自身生平/对方说过的话）。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1")
    assert probe in sys_prompt


# ---------------------------------------------------------------------------
# 4. 反塌陷：**删的是身份，不是情绪**
# ---------------------------------------------------------------------------


async def test_open_gate_keeps_engine_driven_feeling_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开门后引擎情绪入口必须仍在——这是全系统唯一把 e* 传给语言层的通道。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", affect=NEGATIVE)
    assert "你现在的真实心情是" in sys_prompt
    assert affect_label(*NEGATIVE) in sys_prompt
    assert "**应该真实地**影响你的态度、语气和用词" in sys_prompt


async def test_open_gate_keeps_temper_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """脾气段的**行为**指令全留，只摘身份定义——不推翻阶段 15–17「负面时别讨好」的裁定。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", affect=NEGATIVE)
    assert "该不耐烦就不耐烦" in sys_prompt
    assert "你有自尊，不是出气筒" in sys_prompt
    assert "绝不一味道歉、自责、讨好或无限包容" in sys_prompt
    assert "讨好型客服" in sys_prompt, "「客服」反面锚必须保住（它最该起作用的就是负面轮）"


async def test_open_gate_forbids_denying_emotions(monkeypatch: pytest.MonkeyPatch) -> None:
    """反过冲：不许塌陷成「我只是个 AI，没有感情」，也不许身份声明刷屏。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1")
    assert "不要说「我只是个 AI，没有感情」，那是假话" in sys_prompt
    assert "也不必反复声明自己是 AI" in sys_prompt


async def test_open_gate_allows_emotion_without_inventing_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """情绪是真的，但**理由不用编**——防它为情绪虚构人类式由头（累了一天/刚开完会）。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1")
    assert "情绪是真的，但理由不用编" in sys_prompt


# ---------------------------------------------------------------------------
# 5. 关系提示 / 召回引子
# ---------------------------------------------------------------------------


async def test_open_gate_drops_relationship_assertion(monkeypatch: pytest.MonkeyPatch) -> None:
    """「已经比较熟络」是对共同过去的事实断言，依据只是 exposure 计数 → 开门时忽略。"""
    shut = await _system_prompt(monkeypatch, factual=None, relationship_hint="已经比较熟络")
    assert "你和对方目前的关系：已经比较熟络" in shut, "未开门时必须保持原样（零回归）"
    opened = await _system_prompt(monkeypatch, factual="1", relationship_hint="已经比较熟络")
    assert "已经比较熟络" not in opened


async def test_open_gate_replaces_first_person_recall_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """「你还记得以下背景」是第一人称亲历断言 → 换成带出处与可靠度限定的引子。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", retrieved="片段A")
    assert "你还记得以下背景" not in sys_prompt
    assert "不保证相关、不保证完整" in sys_prompt
    assert "说话人是**用户**、不是你自己" in sys_prompt
    assert "片段A" in sys_prompt


# ---------------------------------------------------------------------------
# 6. push 中性死区（成因 C）
# ---------------------------------------------------------------------------


def test_deadzone_off_by_default_reproduces_the_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复现 bug 本体：死区关闭时，`affect_label` 判「平静」的 e* 仍取出暴怒/愤怒/恐惧。

    这条是**证明 bug 存在**的用例——它绿说明缺陷仍可复现，不是护栏。
    """
    monkeypatch.delenv("ZERO_PUSH_NEUTRAL_DEADZONE", raising=False)
    v, a = -0.079, 0.037
    assert affect_label(v, a) == "平静"
    words = suggest_affect_words(v, a, k=6)
    assert "暴怒" in words and "愤怒" in words, "该 e* 应复现「心情平静 vs 用词暴怒」的矛盾"


@pytest.mark.parametrize("val", ["1", "true", "yes", "ON"])
def test_deadzone_env_truthy_enables_global_default(
    monkeypatch: pytest.MonkeyPatch, val: str
) -> None:
    """`ZERO_PUSH_NEUTRAL_DEADZONE` 是三个调用点共用的全局默认——不依赖事实化模式。"""
    monkeypatch.setenv("ZERO_PUSH_NEUTRAL_DEADZONE", val)
    assert suggest_affect_words(-0.079, 0.037, k=6) == []


@pytest.mark.parametrize("val", ["", "0", "false", "off", "no", "2"])
def test_deadzone_env_falsy_keeps_off(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    """⚠ 同 `ZERO_FACTUAL_MODE`：`not in ("", "0")` 式解析会把 `=false` 判成开，逐个钉死。"""
    monkeypatch.setenv("ZERO_PUSH_NEUTRAL_DEADZONE", val)
    assert suggest_affect_words(-0.079, 0.037, k=6) != []


def test_deadzone_explicit_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """调用点显式传 bool 时压过 env（事实化模式即显式 True，需确定性时显式 False）。"""
    monkeypatch.setenv("ZERO_PUSH_NEUTRAL_DEADZONE", "1")
    assert suggest_affect_words(-0.079, 0.037, k=6, neutral_deadzone=False) != []
    monkeypatch.delenv("ZERO_PUSH_NEUTRAL_DEADZONE", raising=False)
    assert suggest_affect_words(-0.079, 0.037, k=6, neutral_deadzone=True) == []


def test_deadzone_on_returns_empty_inside_radius() -> None:
    v, a = -0.079, 0.037
    assert (v**2 + a**2) ** 0.5 < NEUTRAL_RADIUS
    assert suggest_affect_words(v, a, k=6, neutral_deadzone=True) == []


def test_deadzone_on_still_returns_words_outside_radius() -> None:
    """死区只吃「模长趋零」的轮次，真有情绪时照常给词。"""
    words = suggest_affect_words(-0.9, 0.8, k=6, neutral_deadzone=True)
    assert words and "暴怒" in words


def test_deadzone_boundary_is_exclusive() -> None:
    """恰在半径上不算死区（与 affect_label 的 `r < NEUTRAL_RADIUS` 同侧）。"""
    assert suggest_affect_words(NEUTRAL_RADIUS, 0.0, k=3, neutral_deadzone=True) != []
    assert suggest_affect_words(NEUTRAL_RADIUS * 0.99, 0.0, k=3, neutral_deadzone=True) == []


async def test_open_gate_suppresses_push_on_calm_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """开门 + 平静轮 → push 段整段不注入（words 为空则本就不拼）。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", affect=(-0.079, 0.037), push=True)
    assert "用词会偏向这类词的色彩" not in sys_prompt


async def test_open_gate_keeps_push_on_emotional_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """开门 + 真有情绪 → push 段仍在，但用的是去身体化的措辞。"""
    sys_prompt = await _system_prompt(monkeypatch, factual="1", affect=(-0.9, 0.8), push=True)
    assert "用词会偏向这类词的色彩" in sys_prompt
    assert "不自觉地" not in sys_prompt
    assert "如果这些词跟你此刻的心情对不上，以心情为准" in sys_prompt


# ---------------------------------------------------------------------------
# 7. 召回残留（成因 D）：runner 每轮基准必须归零
# ---------------------------------------------------------------------------


async def test_runner_step_zeros_recalled_context_and_facts() -> None:
    """`ConversationSession.step` 每轮基准含 recalled_context/recalled_facts 归零。

    真抓 `graph.ainvoke` 的入参，而不是 `inspect.getsource` 里查字符串——后者
    在注释里提一嘴也能过（既有 test_runner_step_zeros_recalled_episode_ids 就是那种写法）。
    """
    from src.orchestration.runner import ConversationSession, SessionConfig
    from src.orchestration.state import AffectState, Stimulus

    captured: dict[str, Any] = {}

    class _FakeGraph:
        async def ainvoke(self, base: dict[str, Any], config: Any = None) -> AffectState:
            captured.update(base)
            return AffectState()

    session = object.__new__(ConversationSession)
    session.graph = _FakeGraph()
    session.session_id = "s"
    session.user_id = "u"
    session.group_id = "g"
    session.thread_id = "t"
    session.config = SessionConfig()

    await session.step(Stimulus(name="外面还在下雨吗", goal_congruence=0.0, intensity=0.3))

    assert captured["recalled_context"] == [], "未归零 → 未命中轮会读到上一轮的召回"
    assert captured["recalled_facts"] == []
    # 同款条件写入的 disposition：残留会把上一轮倾向错灌进本轮 appraisal 的 prior_mu
    assert captured["recalled_disposition"] is None


async def test_runner_run_zeros_recall_lastvalue_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run()` 与 `step()` 是两条平行入口，归零基准必须同步。

    run() 共享 thread_id 跨 stimulus 持久化（docstring 自述的设计目的），
    MemoryRecallAgent 只在命中时才写 → 未命中轮从 checkpoint 继承上一轮值，
    与 step() 已修的捏造来源同机制。同样真抓 `ainvoke` 入参，不查源码字符串。
    """
    from src.orchestration import runner as runner_mod
    from src.orchestration.state import AffectState, Stimulus

    calls: list[dict[str, Any]] = []

    class _FakeGraph:
        async def ainvoke(self, base: dict[str, Any], config: Any = None) -> AffectState:
            calls.append(dict(base))
            return AffectState()

    monkeypatch.setattr(runner_mod, "build_graph", lambda **kwargs: _FakeGraph())
    monkeypatch.setattr(runner_mod, "build_checkpointer", lambda *args, **kwargs: None)

    await runner_mod.run(
        [
            Stimulus(name="s1", goal_congruence=0.0, intensity=0.3),
            Stimulus(name="s2", goal_congruence=0.0, intensity=0.3),
        ],
        thread_id="t1",
        memory=object(),  # type: ignore[arg-type]  # FakeGraph 不触存储，占位即可
        recall_enabled=True,
        language_enabled=True,
    )

    assert len(calls) == 2
    for base in calls:
        assert base["recalled_context"] == [], "run() 未归零 → 与 step() 已修的残留同机制"
        assert base["recalled_facts"] == []
        assert base["recalled_episode_ids"] == []
        assert base["recalled_disposition"] is None
        assert base["external_priors"] == []


async def test_runner_step_overrides_still_win() -> None:
    """归零是**基准**，state_overrides 仍应能覆盖（与既有三处归零项同语义）。"""
    from src.orchestration.runner import ConversationSession, SessionConfig
    from src.orchestration.state import AffectState, Stimulus

    captured: dict[str, Any] = {}

    class _FakeGraph:
        async def ainvoke(self, base: dict[str, Any], config: Any = None) -> AffectState:
            captured.update(base)
            return AffectState()

    session = object.__new__(ConversationSession)
    session.graph = _FakeGraph()
    session.session_id = "s"
    session.user_id = "u"
    session.group_id = "g"
    session.thread_id = "t"
    session.config = SessionConfig()

    await session.step(
        Stimulus(name="x", goal_congruence=0.0, intensity=0.3),
        state_overrides={"recalled_context": ["显式给的"]},
    )
    assert captured["recalled_context"] == ["显式给的"]


# ---------------------------------------------------------------------------
# 8. chat_driver：召回标签 + 停播种
# ---------------------------------------------------------------------------


def test_recall_tag_defaults_to_original_label() -> None:
    """默认标签逐字不变（零回归）。"""
    from src.memory.types import Fact, Scope
    from src.orchestration.chat_driver import _inject_recalled_as_system

    fact = Fact(
        content="旧事 | precision=40.00",
        scope=Scope.USER,
        key="u",
        sim=1.0,
        valid_at=datetime.now(UTC),
    )
    out = _inject_recalled_as_system([{"role": "user", "content": "hi"}], [fact], 0.0)
    assert out[0]["content"].startswith("（记忆片段）")


def test_recall_tag_can_be_replaced() -> None:
    """事实化标签替换后，出处与人称纠正随每条记忆一起出现在最强位置。"""
    from src.memory.types import Fact, Scope
    from src.orchestration.chat_driver import FACTUAL_RECALL_TAG, _inject_recalled_as_system

    fact = Fact(
        content="旧事 | precision=40.00",
        scope=Scope.USER,
        key="u",
        sim=1.0,
        valid_at=datetime.now(UTC),
    )
    out = _inject_recalled_as_system(
        [{"role": "user", "content": "hi"}], [fact], 0.0, 30.0, FACTUAL_RECALL_TAG
    )
    assert out[0]["content"].startswith(FACTUAL_RECALL_TAG)
    assert "「你说：」= 用户说的" in out[0]["content"]


# ---------------------------------------------------------------------------
# 9. format 安全：新常量不得含裸花括号
# ---------------------------------------------------------------------------


def test_factual_constants_survive_format() -> None:
    """三段拼接过 .format(feeling=…) 不抛；边界段不过 format 故可含任意标点。"""
    combined = _FACTUAL_SYS_HEAD + _FACTUAL_TEMPER_ADDENDUM + _FACTUAL_SYS_TAIL
    assert "{feeling}" in combined
    rendered = combined.format(feeling="平静")
    assert "{" not in rendered and "}" not in rendered


# ---------------------------------------------------------------------------
# 10. 变异验证：证明上面的断言**能红**
# ---------------------------------------------------------------------------


async def test_mutation_gate_actually_switches_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """若门控失效（两种配置产出同一 prompt），本条必红。"""
    shut = await _system_prompt(monkeypatch, factual=None, affect=NEGATIVE)
    opened = await _system_prompt(monkeypatch, factual="1", affect=NEGATIVE)
    assert shut != opened
    assert len(opened) > len(shut), "开门时应更长（多了边界枚举段）"


def test_mutation_default_constants_are_not_the_factual_ones() -> None:
    """防「把默认常量直接改成事实化文本」——那会让所有零回归断言失去意义。"""
    assert _CONVERSE_SYS_HEAD != _FACTUAL_SYS_HEAD
    assert _TEMPER_ADDENDUM != _FACTUAL_TEMPER_ADDENDUM
    assert _CONVERSE_SYS_TAIL != _FACTUAL_SYS_TAIL
    assert "像真人" in _CONVERSE_SYS_TAIL, "默认路径应保持原样（本文件不改默认行为）"
    assert "你是一个**有脾气、有情绪起伏**的人" in _TEMPER_ADDENDUM


def test_mutation_push_deadzone_default_is_off() -> None:
    """死区默认必须关——否则零回归断言（平静轮仍注入 push）会被悄悄改变。"""
    assert suggest_affect_words(-0.079, 0.037, k=6) != []
    assert _PUSH_ADDENDUM != ""
