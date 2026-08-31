"""B1：Dominance→L1 语言风格（2026-08-31 轻量门）——查表纯函数 + 拼装端到端锚点。

裁定要点：三档 ±0.3（复用一致性阈值）；(a′) 数值抑制（|E|≥0.3 或 |A|≥0.3 时不注入，
防与规则 5 强制的 card 语气词双计——纯数值判据无词表，不落 text-predicate-admission）；
门控 ZERO_PERSONA_DOMINANCE_STYLE 默认关；风格句上游合成**不写回 card**。
"""

from __future__ import annotations

import pytest

from src.agents.persona import Persona, _persona_from_dict, dominance_to_style_hint
from src.orchestration.chat_driver import persona_prompt_text

# ── 查表纯函数 ───────────────────────────────────────────────────────────────


def test_high_d_via_oc_injects_assertive() -> None:
    """O/C 主导的高 D（B1 真实生效面）：注入直接自信档。"""
    hint = dominance_to_style_hint(0.42, extraversion=0.0, agreeableness=0.0)
    assert "直截了当" in hint


def test_high_d_via_e_suppressed() -> None:
    """(a′)：E 主导的高 D 被抑制——card 文案按规则 5 必含同方向语气词，再注入=双计。"""
    assert dominance_to_style_hint(0.48, extraversion=0.8, agreeableness=0.0) == ""


def test_high_d_via_a_suppressed() -> None:
    assert dominance_to_style_hint(0.35, extraversion=0.0, agreeableness=0.8) == ""


def test_low_d_via_negative_oc_injects_hedging() -> None:
    """负 O/C 组合的低 D（结构性罕见但可达）：注入商量档。"""
    hint = dominance_to_style_hint(-0.42, extraversion=0.0, agreeableness=0.0)
    assert "商量" in hint


def test_neutral_band_silent() -> None:
    assert dominance_to_style_hint(0.29, extraversion=0.0, agreeableness=0.0) == ""
    assert dominance_to_style_hint(-0.29, extraversion=0.0, agreeableness=0.0) == ""


def test_boundary_inclusive() -> None:
    """阈值 ≥/≤ 含边界（0.3 恰好命中）。"""
    assert dominance_to_style_hint(0.3, extraversion=0.0, agreeableness=0.0) != ""


# ── Persona 派生 ────────────────────────────────────────────────────────────


def test_persona_without_big_five_has_no_dominance() -> None:
    """无 big_five：dominance=None、hint 空——「没配置」与「配置成中性」语义分野。"""
    p = Persona()
    assert p.dominance is None and p.dominance_style_hint == ""


def test_persona_from_big_five_derives_dominance() -> None:
    """O=C=1 卡（E/A 中性）：D=0.42 达阈值且不被 (a′) 抑制 → hint 非空。"""
    p = _persona_from_dict(
        {
            "card": "测试卡",
            "big_five": {"openness": 1.0, "conscientiousness": 1.0},
        }
    )
    assert p.dominance is not None and abs(p.dominance - 0.42) < 1e-9
    assert p.dominance_style_hint != ""


def test_preset_library_currently_all_suppressed() -> None:
    """分布锚点（心理席占比表钉死）：当前 7 卡库 hint 全空——高 D 卡均 E/A 主导被 (a′)
    抑制。此断言红 = 有人加了 O/C 主导新卡（正常，把该卡移出本断言即可）或 (a′) 被改。"""
    import json
    from pathlib import Path

    for path in sorted(
        (Path(__file__).resolve().parent.parent / "personas").glob("*.example.json")
    ):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert _persona_from_dict(data).dominance_style_hint == "", path.name


# ── 拼装端到端锚点（CS 席三条）────────────────────────────────────────────────

_OC_PERSONA = _persona_from_dict(
    {"card": "一张测试人设卡。", "big_five": {"openness": 1.0, "conscientiousness": 1.0}}
)


def test_gate_off_prompt_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """门关（默认）：即便 hint 非空，注入文本恰为 card 字节原文——端到端零回归。"""
    monkeypatch.delenv("ZERO_PERSONA_DOMINANCE_STYLE", raising=False)
    assert persona_prompt_text(_OC_PERSONA) == _OC_PERSONA.card


def test_gate_on_empty_hint_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    neutral = Persona(card="只有卡。")
    assert persona_prompt_text(neutral) == "只有卡。"


def test_gate_on_hint_appended_not_written_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    text = persona_prompt_text(_OC_PERSONA)
    assert text.startswith(_OC_PERSONA.card) and "直截了当" in text
    assert "直截了当" not in _OC_PERSONA.card  # 不写回 card（预设卡锚点语义不动）


def test_gate_on_card_empty_hint_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    p = _persona_from_dict({"big_five": {"openness": 1.0, "conscientiousness": 1.0}})
    assert persona_prompt_text(p) == p.dominance_style_hint
