"""fear 跨层「面部泄漏」联合锚点（2026-08-31 轻量门复裁：维持表情层不加门）。

裁定：标签层保守归类（fear 门关 → rage）与表情层 fear-AU 点亮（负 coping 连续映射）
**同时成立是有意行为**——「低控制感面部泄漏」（Susskind 2008 感知准备动作，先于归类）。
本文件把这个跨层不变式钉进同一断言：将来任何一侧单独漂移（有人给表情层悄悄加门、
或放松标签层保守回退）都会在此变红——此前两侧锚点各在一处、无人看守联合关系。
"""

from __future__ import annotations

from src.agents.affect_math import decode_channels
from src.agents.emotion_lexicon import motivational_system

# 生效组合：coping 门开语义下的负控制感（-v,+a 象限）
_V, _A, _CP = -0.5, 0.5, -0.5


def test_fear_gate_off_label_rage_but_fear_aus_light_up() -> None:
    """门关 + 负 coping：标签保守回退 rage **且** 表情层恐惧 AU 点亮——有意不一致。"""
    label = motivational_system(_V, _A, coping_potential=_CP, fear_domain_enabled=False)
    assert label == "rage", "标签层保守回退是 2026-07-21 fear 默认关裁决的核心"
    facs = decode_channels((_V, _A), coping_potential=_CP, facs_extended=True)["facs_au"]
    assert facs["AU01"] > 0.0 and facs["AU02"] > 0.0 and facs["AU20"] > 0.0, (
        "表情层泄漏是 2026-07-21 B-facs-fear 正交裁定 + 2026-08-31 复裁的有意行为；"
        "若本断言红=有人给表情层加了门，须先推翻两次议会裁定"
    )


def test_fear_gate_on_label_and_aus_agree() -> None:
    """门开对照：标签与表情同向（fear + AU 点亮），钉死「不一致仅存在于门关侧」。"""
    label = motivational_system(_V, _A, coping_potential=_CP, fear_domain_enabled=True)
    assert label == "fear"
    facs = decode_channels((_V, _A), coping_potential=_CP, facs_extended=True)["facs_au"]
    assert facs["AU20"] > 0.0


def test_positive_coping_no_leakage() -> None:
    """正 coping（愤怒向）：fear-AU 恒 0、AU23 点亮——泄漏只在负向，方向锚点。"""
    facs = decode_channels((_V, _A), coping_potential=0.5, facs_extended=True)["facs_au"]
    assert facs["AU20"] == 0.0 and facs["AU01"] == 0.0
    assert facs["AU23"] > 0.0
