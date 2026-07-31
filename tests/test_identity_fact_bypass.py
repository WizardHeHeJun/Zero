"""身份自陈写入旁路单测（PRP identity-fact-write-bypass · 议会 2026-07-30 D1）。

覆盖 prp.md 验证清单 V1/V2/V3/V5/V6/V8/V9/V10/V11（V4 双臂 E2E 与 V7 召回可达在
test_identity_fact_e2e.py）。

⚠ 本文件的阈值与样本集来自 `tests/fixtures_identity_disclosure.py`，
那是**先于实现单独成 commit** 的预注册产物，不得在这里就地放宽。
"""

from __future__ import annotations

import random

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration import supervisor as sup
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import (
    IDENTITY_MEMORY_PRECISION,
    SupervisorAgent,
    _env_flag_directionless,
    _is_identity_disclosure,
)
from tests.fixtures_hundred_turn_script import (
    HUNDRED_TURN_SCRIPT,
    IDENTITY_TURNS,
    NON_IDENTITY_HOLDOUT,
)
from tests.fixtures_identity_disclosure import (
    IDENTITY_NEGATIVES,
    IDENTITY_POSITIVES,
    MAX_HOLDOUT_HITS,
    MAX_NEGATIVE_HITS,
    MIN_POSITIVE_HITS,
)


class _RecordingStore:
    """记录 add_episode 调用的 fake SemanticStore（仿 test_episodic_memory 的同名件）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def add_episode(
        self,
        content: str,
        *,
        scope: str,
        key: str,
        valid_at: object = None,
        embed_text: str | None = None,
    ) -> None:
        self.calls.append({"content": content, "scope": scope, "key": key})

    async def search(self, *args: object, **kwargs: object) -> list:
        return []


def _make_state(
    text: str,
    *,
    precision: float = 8.56,
    rpe: float = 0.0,
    user_id: str = "u1",
) -> AffectState:
    """构造一轮 supervisor 输入。默认 precision/rpe 取 100 轮实测的身份轮真实值。

    默认 salience = 8.56 × 0.0 = 0.0 < 门 0.15 ⇒ 主门不写，从而单独检验旁路。
    """
    return AffectState(
        stimulus=Stimulus(name=text[:40], text=text, goal_congruence=0.0, intensity=0.0),
        affect_sample=(0.0, 0.0),
        affect_precision=precision,
        rpe=rpe,
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# V1 四段判据 —— 各段独立单测
# ---------------------------------------------------------------------------


def test_segment1_requires_self_reference() -> None:
    """段 1：非自指主语不命中（第三方事实）。"""
    assert _is_identity_disclosure("我朋友是医生") is None
    assert _is_identity_disclosure("我妈是退休教师") is None
    assert _is_identity_disclosure("他是做后端的") is None


def test_segment1_allows_leading_filler() -> None:
    """段 1：不锚 ^，允许「对了我叫…」这类前缀。"""
    assert _is_identity_disclosure("对了我叫小林") == ("name", "小林")


def test_segment2_maps_predicate_to_kind() -> None:
    """段 2：谓词决定属性类型。"""
    assert _is_identity_disclosure("我叫林川") == ("name", "林川")
    assert _is_identity_disclosure("我是老师") == ("occupation", "老师")


def test_segment3_object_must_be_in_closed_set() -> None:
    """段 3：宾语走闭集，不在职业表内的一律不命中。

    ⚠ 这一条直接钉住首版的失败：若把判据写成「长度 + 字符类」，
    「我是做后端开发的」与「我是做什么工作的」在长度与字符类上完全同构、无法区分。
    """
    assert _is_identity_disclosure("我是做后端开发的") == ("occupation", "后端")
    assert _is_identity_disclosure("我是真的累了") is None
    assert _is_identity_disclosure("我在等一个结果") is None


def test_segment3_stopwords_reject_pronoun_object() -> None:
    """段 3b：宾语首位是停用词/代词即放行（「我叫你别管」的「叫」是使役不是自陈）。"""
    assert _is_identity_disclosure("我叫你别管") is None
    assert _is_identity_disclosure("我是说你别急") is None


def test_segment4_excludes_questions() -> None:
    """段 4：疑问句一律放行——真实语料里唯一实际发生的误报形态。"""
    assert _is_identity_disclosure("我是做什么工作的") is None
    assert _is_identity_disclosure("考考你，我叫什么名字") is None
    assert _is_identity_disclosure("我是谁") is None


def test_empty_text_is_none() -> None:
    assert _is_identity_disclosure("") is None


# ---------------------------------------------------------------------------
# V1/V2 预注册样本集与真实留出集
# ---------------------------------------------------------------------------


def test_positive_samples_meet_preregistered_count() -> None:
    hits = sum(
        1
        for text, kind in IDENTITY_POSITIVES
        if (got := _is_identity_disclosure(text)) is not None and got[0] == kind
    )
    assert hits >= MIN_POSITIVE_HITS, (
        f"正样本命中 {hits}/{len(IDENTITY_POSITIVES)}，低于预注册阈值 {MIN_POSITIVE_HITS}。"
        "⚠ 应改判据，不得放宽本阈值或删改样本集。"
    )


def test_negative_samples_meet_preregistered_count() -> None:
    false_positives = [t for t in IDENTITY_NEGATIVES if _is_identity_disclosure(t) is not None]
    assert len(false_positives) <= MAX_NEGATIVE_HITS, f"负样本误报：{false_positives}"


def test_real_holdout_has_zero_false_positives() -> None:
    """真实留出集（100 轮实跑里的全部非身份语句）误报数须为 0。

    这是本特性唯一有统计意义的那条：n≈90、0 误报 → FP 率 95% 上界约 3.3%。
    """
    false_positives = [t for t in NON_IDENTITY_HOLDOUT if _is_identity_disclosure(t) is not None]
    assert len(false_positives) <= MAX_HOLDOUT_HITS, f"留出集误报：{false_positives}"
    assert len(NON_IDENTITY_HOLDOUT) >= 50, "留出集过小，本用例失去判别力"


# ---------------------------------------------------------------------------
# V5 全语料扫描（绕开去重，直接检验判据本身）
# ---------------------------------------------------------------------------


def test_full_script_scan_hits_exactly_identity_turns() -> None:
    hits = {i for i, t in enumerate(HUNDRED_TURN_SCRIPT, start=1) if _is_identity_disclosure(t)}
    assert len(hits) < len(HUNDRED_TURN_SCRIPT), "否则本用例没在测东西（判据命中了所有句子）"
    assert hits == set(IDENTITY_TURNS), f"命中轮次 {sorted(hits)} != 期望 {sorted(IDENTITY_TURNS)}"


# ---------------------------------------------------------------------------
# V3 变异测试（双向义务）
# ---------------------------------------------------------------------------


def _assert_identity_written(store: _RecordingStore, expect_substr: str) -> None:
    """被检验的断言本身——变异测试要驱动它，而不是在旁边另算一个不等式。"""
    assert any(expect_substr in c["content"] for c in store.calls), (
        f"未写入含「{expect_substr}」的 episode；实际 {len(store.calls)} 条"
    )


async def test_mutation_a1_assertion_itself_goes_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """(a1) 让判据恒返 None → 上面那条辅助断言必须抛 AssertionError。"""
    monkeypatch.setattr(sup, "_is_identity_disclosure", lambda _t: None)
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川，做后端开发的"))
    with pytest.raises(AssertionError):
        _assert_identity_written(store, "我叫林川")


def test_mutation_a2_loosened_object_segment_breaks_negatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a2) 实现级变异：把 **shipped 的** 职业正则换成「字符类捕获」的宽松版，负样本立刻被误收。

    ⚠ 必须 monkeypatch 生产模块里的那个对象、再调真 `_is_identity_disclosure`——
    首版是在测试里现造一份宽松正则自己 search，那验证的是"这类正则会误收"这个通用陈述，
    与 shipped 代码零耦合（code-reviewer WARN-3）。
    """
    import re

    baseline = [t for t in IDENTITY_NEGATIVES if _is_identity_disclosure(t) is not None]
    assert baseline == [], f"变异前不应有误报，实际 {baseline}"

    monkeypatch.setattr(
        sup,
        "_IDENTITY_JOB_RE",
        re.compile(rf"我{sup._SELF_ADVERB}(?:是|在|做)[^，,。.！!？?]{{0,10}}?([一-龥]{{2,4}})"),
    )
    caught = [t for t in IDENTITY_NEGATIVES if _is_identity_disclosure(t) is not None]
    assert caught, (
        "把职业段从闭集换成字符类后本应误收若干负样本；若仍为空，说明该段并非负样本被拒的原因，"
        "本用例测的不是它宣称要测的东西"
    )


def test_mutation_a2_without_question_segment_false_positive_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a2) 实现级变异：把段 4 换成永不匹配，真实语料里那条假阳必须回来。

    这才证明段 4 在起判别作用——而不是"恰好没被触发"。
    """
    import re

    # ⚠ 必须挑一个**段 3 会放行、只有段 4 能拦**的例子，否则测不出段 4 的判别力。
    # 反例记录：「我是做什么工作的」去掉段 4 后仍是 None——真正拦它的是段 3 的职业闭集
    # （"工作"不在职业表里），段 4 对它只是冗余的第二道防线。首版用它做变异体，红了才发现假设错。
    text = "我是老师吗"
    assert _is_identity_disclosure(text) is None, "有段 4 时疑问句应被排除"

    monkeypatch.setattr(sup, "_QUESTION_RE", re.compile(r"(?!x)x"))  # 永不匹配
    assert _is_identity_disclosure(text) == ("occupation", "老师"), (
        "去掉疑问排除后这条应当被段 1-3 收进来；若仍为 None，说明段 4 并非它被排除的原因，"
        "本用例测的不是它宣称要测的东西"
    )


def test_mutation_b_correct_implementation_is_stably_green() -> None:
    """(b) 正确实现上恒绿：正样本全集打乱 20 轮，结果稳定。

    (b) 比 (a) 更要紧——「在正确实现上会红的断言」会在落地时被调松。
    """
    rng = random.Random(20260731)
    samples = list(IDENTITY_POSITIVES)
    for _ in range(20):
        rng.shuffle(samples)
        for text, kind in samples:
            got = _is_identity_disclosure(text)
            assert got is not None and got[0] == kind, f"不稳定：{text} → {got}"


# ---------------------------------------------------------------------------
# V6 节流（有判别力的界）
# ---------------------------------------------------------------------------


async def test_no_write_for_non_identity_low_salience_turns() -> None:
    """(a) 在非身份语料上，salience 与 commitment 都不成立时不得发生任何写入。

    ⚠ 替代首版「每轮写入 ≤1 次」——那条按构造恒真（写入唯一调用点在唯一的 if 块内）。
    """
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    plain = [t for t in NON_IDENTITY_HOLDOUT if not sup._is_commitment(t)][:60]
    assert len(plain) >= 50, "非承诺类语料不足 50 条，本用例失去判别力"
    for text in plain:
        await agent(_make_state(text))
    assert store.calls == [], f"非身份低 salience 轮次不应写入，实际写了 {len(store.calls)} 条"


async def test_bypass_extra_writes_bounded_by_distinct_facts() -> None:
    """(b) 旁路带来的额外写入总数 ≤ 不同身份事实条数（由三元组去重兜底）。"""
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    for _ in range(8):
        await agent(_make_state("我叫林川"))
        await agent(_make_state("我是做后端开发的"))
    assert len(store.calls) == 2, f"两个身份事实各写一次，实际 {len(store.calls)} 条"


# ---------------------------------------------------------------------------
# V8 去重三组
# ---------------------------------------------------------------------------


async def test_dedup_same_fact_written_once() -> None:
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川"))
    await agent(_make_state("我叫林川"))
    assert len(store.calls) == 1


async def test_correction_writes_second_episode() -> None:
    """改口必须写第二条——按属性类型一次性去重会吞掉它，违 memory-rules #4。"""
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫李川"))
    await agent(_make_state("我叫林川"))
    assert len(store.calls) == 2, "同类型不同实体是新事实，须照写（时序失效交给记忆层）"


async def test_dedup_is_scoped_per_user() -> None:
    """同一 agent 实例下不同 user 各自写入，不得跨用户串味（memory-rules #2）。"""
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川", user_id="userA"))
    await agent(_make_state("我叫周野", user_id="userB"))
    assert len(store.calls) == 2
    assert {c["key"] for c in store.calls} == {"userA", "userB"}


async def test_registration_happens_even_when_main_gate_fired() -> None:
    """登记时机：高唤醒身份自陈经主门写入后也要打标，否则平静复述会二次写入。"""
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    # 高 salience：precision×|rpe| = 8.56×0.9 远超门 0.15，主门自己就会写
    await agent(_make_state("我叫林川", rpe=0.9))
    assert len(store.calls) == 1
    await agent(_make_state("我叫林川"))  # 平静复述
    assert len(store.calls) == 1, "主门触发的那次也应登记，复述不得二次写入"


# ---------------------------------------------------------------------------
# V9 重叠命中：同一轮只写一次
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rpe"),
    [
        ("我叫林川，明天下午两点见", 0.0),  # commitment × identity
        ("我叫林川", 0.9),  # salient × identity
        ("我叫林川，明天下午两点见", 0.9),  # 三者同时
    ],
)
async def test_overlapping_hits_write_exactly_once(text: str, rpe: float) -> None:
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state(text, rpe=rpe))
    assert len(store.calls) == 1, f"重叠命中应仅写一次，实际 {len(store.calls)}"


# ---------------------------------------------------------------------------
# V10 env 取值表（方向无关解析）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("1", True),
        ("true", True),
        (" True ", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("no", False),
        ("enabled", True),  # 未识别 → 回落 default，而非静默翻成 False
    ],
)
def test_env_flag_is_direction_agnostic(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool
) -> None:
    """默认 True 的旗标必须方向无关——只判真值集的写法会让未识别值静默关掉开关。"""
    name = "ZERO_IDENTITY_FACT_BYPASS"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert _env_flag_directionless(name, True) is expected


def test_env_flag_matches_mcp_side_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """跨侧一致：与 MCP 边界的同款解析逐值相等（该 pitfall 的处方后半）。"""
    mcp_flag = pytest.importorskip("src.mcp_server.server")._env_flag
    name = "ZERO_IDENTITY_FACT_BYPASS"
    for raw in ("", "   ", "1", "true", " True ", "0", "false", "off", "enabled", "yes", "no"):
        monkeypatch.setenv(name, raw)
        assert _env_flag_directionless(name, True) is mcp_flag(name, True), (
            f"取值 {raw!r} 两侧不一致"
        )


# ---------------------------------------------------------------------------
# V11 门控关闭 / scope / 精度下限
# ---------------------------------------------------------------------------


async def test_bypass_disabled_is_zero_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_IDENTITY_FACT_BYPASS", "0")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川，做后端开发的"))
    assert store.calls == [], "旗标关闭时行为须与改动前逐字一致（低 salience 不写）"


async def test_identity_episode_uses_user_scope() -> None:
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川"))
    assert store.calls[0]["scope"] == Scope.USER


async def test_identity_episode_precision_clears_recall_gate() -> None:
    """架构决策 F：身份 episode 的 precision 须使 Hill 归一后高于召回注入门默认 0.5。"""
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    await agent(_make_state("我叫林川"))
    content = store.calls[0]["content"]
    assert f"precision={IDENTITY_MEMORY_PRECISION:.2f}" in content, content
    normalized = IDENTITY_MEMORY_PRECISION / (IDENTITY_MEMORY_PRECISION + 30.0)
    assert normalized > 0.5, "否则写进去也进不了 LLM 注意力预算"
