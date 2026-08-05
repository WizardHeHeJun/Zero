"""舞台说明 → 行为意图的路由：剥离器新返回值 + 端到端映射。

关键不变式：**报告的片段 == 真正从可见文本里拿掉的东西**。若两者脱节，就会出现
「舞台说明还留在文本里、同时又驱动了形象」的双重表达。
"""

from __future__ import annotations

from src.agents.behavior_intent import merge_intents, stage_direction_intents
from src.agents.language_openai import (
    strip_stage_directions,
    strip_stage_directions_with_segments,
)


def test_cleaned_text_matches_legacy_function() -> None:
    """新函数的文本部分与既有 `strip_stage_directions` 逐字一致（零回归的直接断言）。"""
    samples = [
        "（点了点头）我知道了。",
        "（1）第一点。（2）第二点。",
        "我这边（没有时钟），看不到时间。",
        "他笑了笑（无奈地），没再说什么。",
        "普通一句话，没有括号。",
        "（笑）",
    ]
    for text in samples:
        cleaned, _ = strip_stage_directions_with_segments(text)
        assert cleaned == strip_stage_directions(text), text


def test_segments_are_reported_when_stripped() -> None:
    cleaned, segments = strip_stage_directions_with_segments("（点了点头）我知道了。")
    assert "点" not in cleaned
    assert segments == ["点了点头"]


def test_kept_parentheses_are_not_reported() -> None:
    """排除表保住的括号（编号/免责语）没被拿掉 → 不得出现在片段里。"""
    _, segments = strip_stage_directions_with_segments("（1）第一点。")
    assert segments == []
    _, segments2 = strip_stage_directions_with_segments("我这边（没有时钟），看不到。")
    assert segments2 == []


def test_full_strip_fallback_reports_no_segments() -> None:
    """全剥空 → 回退原文 → 文本实际未变 ⇒ 必须不报告片段。

    否则「（笑）」会既留在可见回复里、又触发一次微笑动作 = 双重表达。
    """
    cleaned, segments = strip_stage_directions_with_segments("（笑）")
    assert cleaned == "（笑）"  # 回退原文
    assert segments == []


def test_end_to_end_stage_direction_becomes_behavior() -> None:
    """端到端：模型自发写的动作 → 剥出片段 → 映射成 12 词行为。"""
    cleaned, segments = strip_stage_directions_with_segments("（皱了皱眉）这事有点麻烦。")
    intents = merge_intents([], stage_direction_intents(segments))
    assert "皱" not in cleaned
    assert [i.name for i in intents] == ["brow_furrow"]


def test_end_to_end_physical_claim_yields_no_behavior() -> None:
    """🛑 边界：物理世界宣称即便被剥掉，也**不得**变成行为。

    有了皮套「点头」成为真能力，「关灯」依然不是——闭集白名单是这条边界的执行机制。
    """
    _, segments = strip_stage_directions_with_segments("（我帮你把灯关了）好点没？")
    assert merge_intents([], stage_direction_intents(segments)) == []
