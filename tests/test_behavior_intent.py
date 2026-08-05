"""行为意图抽取单测：词法判定、舞台说明映射、以及**闭集安全边界**。

重点覆盖两类容易写错、且错了后果不对称的地方：
- 否定句上点头（语义反向，比不动更糟）；
- 物理世界行动宣称漏进闭集（数字人重新开始宣称它做不到的事）。
"""

from __future__ import annotations

from src.agents.behavior_intent import (
    BEHAVIOR_VOCABULARY,
    BehaviorIntent,
    lexical_intents,
    merge_intents,
    stage_direction_intents,
)


def _names(intents: list[BehaviorIntent]) -> list[str]:
    return [i.name for i in intents]


# ── ③a 词法 ────────────────────────────────────────────────────────────────


def test_affirmation_yields_nod() -> None:
    assert "nod" in _names(lexical_intents("嗯，是这样的。"))


def test_negation_yields_shake_not_nod() -> None:
    """否定句必须摇头而**不是**点头——在否定句上点头是语义反向，比不动更糟。

    「不是」「没有」内含「是」「有」，判定顺序写反就会在这里翻车。
    """
    for text in ("不是吧，我没有说过。", "这不对。", "我不能这么做。"):
        names = _names(lexical_intents(text))
        assert "shake" in names, text
        assert "nod" not in names, text


def test_question_yields_head_tilt_with_direction() -> None:
    intents = lexical_intents("你说的是哪一次？")
    tilt = [i for i in intents if i.name == "head_tilt"]
    assert tilt and tilt[0].direction is not None


def test_emphasis_yields_brow_raise() -> None:
    assert "brow_raise" in _names(lexical_intents("这个真的很重要！"))


def test_empty_reply_yields_nothing() -> None:
    assert lexical_intents("") == []
    assert lexical_intents("   ") == []


def test_plain_statement_yields_nothing() -> None:
    """无明确词形证据时不产出——取向是宁漏勿误。"""
    assert lexical_intents("今天天气我看不到。") == []


def test_lexical_output_is_capped() -> None:
    intents = lexical_intents("对啊，真的一定要这样吗？！")
    assert len(intents) <= 2


# ── ③b 舞台说明路由 ────────────────────────────────────────────────────────


def test_stage_direction_maps_to_behavior() -> None:
    """模型自发产出的「点了点头」应转成行为，而不是被白白删掉。"""
    assert _names(stage_direction_intents(["点了点头"])) == ["nod"]
    assert _names(stage_direction_intents(["无奈地笑了笑"])) == ["smile"]
    assert _names(stage_direction_intents(["皱了皱眉"])) == ["brow_furrow"]


def test_physical_world_claims_are_discarded() -> None:
    """🛑 安全边界：对物理世界的行动宣称映射不进闭集 → 必须丢弃。

    阶段 63 删舞台说明的根因是模型虚构自己没有的具身能力。有了皮套，「点头」成为真能力，
    「关灯」依然不是。这条红了 = 数字人又能宣称它做不到的事。
    """
    claims = ["我帮你关灯了", "走过去把窗户打开", "递给你一杯水", "站起来走到门口"]
    assert stage_direction_intents(claims) == []


def test_one_segment_yields_at_most_one_behavior() -> None:
    """「笑着点头」只出一个，避免两条抢同一通道。"""
    assert len(stage_direction_intents(["笑着点了点头"])) == 1


def test_stage_intents_deduplicate() -> None:
    assert _names(stage_direction_intents(["点头", "又点了点头"])) == ["nod"]


# ── 合并与闭集守卫 ─────────────────────────────────────────────────────────


def test_stage_takes_precedence_over_lexical() -> None:
    """舞台说明是模型的显式表达意图，比词法推断更可信 → 同名时保留它。"""
    lexical = [BehaviorIntent("nod", intensity=0.5, source="lexical")]
    stage = [BehaviorIntent("nod", intensity=0.9, source="stage")]
    merged = merge_intents(lexical, stage)
    assert len(merged) == 1
    assert merged[0].source == "stage"


def test_merge_rejects_out_of_vocabulary() -> None:
    """闭集守卫：即便有人往映射表里塞了新词，合并这一关也不放行。

    这是边界的**第二道**——防「改映射表就能绕过安全边界」。
    """
    rogue = [BehaviorIntent("turn_off_light", intensity=1.0, source="stage")]
    assert merge_intents([], rogue) == []


def test_merge_respects_limit() -> None:
    stage = [BehaviorIntent(n, source="stage") for n in ("nod", "smile", "lean_in", "blink")]
    assert len(merge_intents([], stage, limit=3)) == 3


def test_all_mapped_behaviors_are_in_vocabulary() -> None:
    """映射表产出的每个名字都必须在 12 词闭集内（防手误引入无效词）。"""
    samples = [
        "点头",
        "摇头",
        "歪头",
        "挑眉",
        "皱眉",
        "瞪大眼睛",
        "笑",
        "凑近",
        "后仰",
        "看向窗外",
        "眨眨眼",
        "轻轻晃",
    ]
    for intent in stage_direction_intents(samples):
        assert intent.name in BEHAVIOR_VOCABULARY
