"""大五（OCEAN）→ PAD 气质映射测试（框架改进 B7c / 议会 A-P3-E）。

系数来自 Mehrabian (1996) 线性回归（DOI:10.1080/00049539608259510），第五维原文为情绪稳定性
S=−N，代入后入参用大五标准 Neuroticism。默认不提供大五 → 现有中性 setpoint，零回归。

T5（议会 2026-07-02 Follow-up 3）：big_five_to_va_coupling 数值 + persona 推导 + 显式优先 +
贯通链路（persona 带 big_five → build_chat_driver → session.config.va_coupling_*
优先于 env；无 persona va_coupling 时才回落 env）。
"""

from __future__ import annotations

import pytest

from src.agents.persona import Persona, _persona_from_dict, big_five_to_pad


def test_big_five_neutral_maps_to_neutral() -> None:
    """全中性大五（全 0）→ PAD 全 0。"""
    assert big_five_to_pad(0.0, 0.0, 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_big_five_coefficients_extraversion() -> None:
    """E=1 → (0.21, 0, 0.60)：核对 Mehrabian 系数（pleasure/arousability/dominance）。"""
    p, a, d = big_five_to_pad(0.0, 0.0, 1.0, 0.0, 0.0)
    assert abs(p - 0.21) < 1e-9
    assert abs(a - 0.0) < 1e-9
    assert abs(d - 0.60) < 1e-9


def test_big_five_coefficients_agreeableness() -> None:
    """A=1 → (0.59, 0.30, -0.32)。"""
    p, a, d = big_five_to_pad(0.0, 0.0, 0.0, 1.0, 0.0)
    assert abs(p - 0.59) < 1e-9
    assert abs(a - 0.30) < 1e-9
    assert abs(d - (-0.32)) < 1e-9


def test_big_five_neuroticism_sign() -> None:
    """N=1 → 低愉悦(-0.19)、高唤醒性(+0.57)：符号自洽（S=-N 代入）。"""
    p, a, d = big_five_to_pad(0.0, 0.0, 0.0, 0.0, 1.0)
    assert abs(p - (-0.19)) < 1e-9
    assert abs(a - 0.57) < 1e-9
    assert abs(d - 0.0) < 1e-9


def test_big_five_output_clamped() -> None:
    """极端大五组合仍 clamp 到 [-1,1]。"""
    p, a, d = big_five_to_pad(1.0, 1.0, 1.0, 1.0, 1.0)
    assert -1.0 <= p <= 1.0 and -1.0 <= a <= 1.0 and -1.0 <= d <= 1.0


def test_persona_big_five_derives_setpoint() -> None:
    """persona JSON 给 big_five（dict）→ setpoint 由方程推导。"""
    persona = _persona_from_dict({"big_five": {"extraversion": 1.0, "agreeableness": 1.0}})
    assert abs(persona.setpoint[0] - (0.21 + 0.59)) < 1e-9  # pleasure
    assert abs(persona.setpoint[1] - 0.30) < 1e-9  # arousability


def test_persona_big_five_list_form() -> None:
    """big_five 也接受 OCEAN 5 元素列表。"""
    persona = _persona_from_dict({"big_five": [0.0, 0.0, 1.0, 0.0, 0.0]})
    assert abs(persona.setpoint[0] - 0.21) < 1e-9


def test_explicit_setpoint_wins_over_big_five() -> None:
    """显式 setpoint 优先于 big_five。"""
    persona = _persona_from_dict({"setpoint": [0.5, -0.5], "big_five": {"extraversion": 1.0}})
    assert persona.setpoint == (0.5, -0.5)


def test_no_big_five_neutral_zero_regression() -> None:
    """不给大五也不给 setpoint → 中性默认（零回归）。"""
    persona = _persona_from_dict({"card": "x"})
    assert persona.setpoint == Persona().setpoint


# ---------------------------------------------------------------------------
# T5：big_five_to_va_coupling 数值 + persona 推导 + 显式优先 + 贯通链路
# ---------------------------------------------------------------------------


def test_va_coupling_neutral_extraversion() -> None:
    """E=0 → (pos=0.50, neg=0.65)（baseline，negativity bias offset 0.15 维持）。"""
    import pytest

    from src.agents.persona import big_five_to_va_coupling

    pos, neg = big_five_to_va_coupling(0.0)
    assert pos == pytest.approx(0.50)
    assert neg == pytest.approx(0.65)


def test_va_coupling_high_extraversion() -> None:
    """E=+1 → (pos=0.60, neg=0.75)（两侧各 +0.10，negativity bias offset 恒 0.15）。"""
    import pytest

    from src.agents.persona import big_five_to_va_coupling

    pos, neg = big_five_to_va_coupling(1.0)
    assert pos == pytest.approx(0.60)
    assert neg == pytest.approx(0.75)


def test_va_coupling_low_extraversion() -> None:
    """E=-1 → (pos=0.40, neg=0.55)（两侧各 -0.10，negativity bias offset 恒 0.15）。"""
    import pytest

    from src.agents.persona import big_five_to_va_coupling

    pos, neg = big_five_to_va_coupling(-1.0)
    assert pos == pytest.approx(0.40)
    assert neg == pytest.approx(0.55)


def test_va_coupling_neg_always_greater_than_pos() -> None:
    """任意 E 值，neg > pos（negativity bias 恒成立）。"""
    from src.agents.persona import big_five_to_va_coupling

    for e in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        pos, neg = big_five_to_va_coupling(e)
        assert neg > pos, f"E={e} 时应有 neg > pos，但 pos={pos:.4f} neg={neg:.4f}"


def test_va_coupling_clamp_boundaries() -> None:
    """极端 E 值不超出 clamp 范围：pos∈[0.35,0.70]，neg∈[0.50,0.85]。"""
    from src.agents.persona import big_five_to_va_coupling

    # 远超范围的 E 值（如 ±100）被 clamp 收口
    for e_extreme in [-100.0, 100.0]:
        pos, neg = big_five_to_va_coupling(e_extreme)
        assert 0.35 <= pos <= 0.70, f"pos={pos:.4f} 超出 [0.35,0.70]，E={e_extreme}"
        assert 0.50 <= neg <= 0.85, f"neg={neg:.4f} 超出 [0.50,0.85]，E={e_extreme}"


def test_persona_from_dict_big_five_derives_va_coupling() -> None:
    """persona JSON 带 big_five（给 extraversion）且无显式 va_coupling → 由推导填充（非 None）。"""
    from src.agents.persona import big_five_to_va_coupling

    e_val = 0.5
    persona = _persona_from_dict({"big_five": {"extraversion": e_val}})
    expected_pos, expected_neg = big_five_to_va_coupling(e_val)
    # 推导值不为 None
    assert persona.va_coupling_pos is not None, "va_coupling_pos 应由 big_five 推导，不应为 None"
    assert persona.va_coupling_neg is not None, "va_coupling_neg 应由 big_five 推导，不应为 None"
    # 推导值等于 big_five_to_va_coupling(E)
    import pytest

    assert persona.va_coupling_pos == pytest.approx(expected_pos)
    assert persona.va_coupling_neg == pytest.approx(expected_neg)


def test_persona_explicit_va_coupling_wins_over_big_five() -> None:
    """显式 va_coupling_pos/neg 优先于 big_five 推导（不被推导覆盖）。"""
    import pytest

    persona = _persona_from_dict(
        {
            "big_five": {"extraversion": 1.0},  # 推导值：pos=0.60, neg=0.75
            "va_coupling_pos": 0.45,  # 显式值：应优先
            "va_coupling_neg": 0.80,  # 显式值：应优先
        }
    )
    assert persona.va_coupling_pos == pytest.approx(0.45), (
        f"显式 va_coupling_pos 应优先，实际 {persona.va_coupling_pos}"
    )
    assert persona.va_coupling_neg == pytest.approx(0.80), (
        f"显式 va_coupling_neg 应优先，实际 {persona.va_coupling_neg}"
    )


def test_persona_explicit_pos_only_neg_derived() -> None:
    """只显式给 va_coupling_pos，neg 由 big_five 推导（混合优先级）。"""
    import pytest

    from src.agents.persona import big_five_to_va_coupling

    e_val = 0.0
    persona = _persona_from_dict(
        {
            "big_five": {"extraversion": e_val},
            "va_coupling_pos": 0.45,  # 只显式给 pos
            # va_coupling_neg 未给 → 应由 big_five 推导
        }
    )
    _, expected_neg = big_five_to_va_coupling(e_val)
    assert persona.va_coupling_pos == pytest.approx(0.45), "显式 pos 应保留"
    assert persona.va_coupling_neg == pytest.approx(expected_neg), (
        f"neg 应由 big_five 推导，期望 {expected_neg}，实际 {persona.va_coupling_neg}"
    )


def test_persona_no_big_five_no_va_coupling_is_none() -> None:
    """无 big_five 且无显式 va_coupling → persona.va_coupling_pos/neg 均为 None（零回归）。"""
    persona = _persona_from_dict({"card": "test"})
    assert persona.va_coupling_pos is None, "无 big_five 时 va_coupling_pos 应为 None（零回归）"
    assert persona.va_coupling_neg is None, "无 big_five 时 va_coupling_neg 应为 None（零回归）"


def test_build_chat_driver_persona_big_five_va_coupling_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """persona JSON 带 big_five → build_chat_driver 推导 va_coupling，覆盖 env 里的值。

    链路：persona_file.big_five → big_five_to_va_coupling(E) → session.config.va_coupling_*
    即使同时设了 ZERO_VA_COUPLING_POS/NEG env，persona 推导值优先（persona > env）。
    """
    import json
    import os as _os
    import tempfile

    import pytest

    from src.agents.persona import big_five_to_va_coupling
    from src.orchestration.chat_driver import build_chat_driver

    e_val = 0.5
    data = {
        "name": "test-t5",
        "card": "test card",
        "big_five": {"extraversion": e_val},
        # 无显式 va_coupling_pos/neg → 由 big_five 推导
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = f.name
    try:
        monkeypatch.setenv("ZERO_PERSONA_FILE", tmp_path)
        # 设置 env 里的 va_coupling 值（应被 persona 推导值覆盖）
        monkeypatch.setenv("ZERO_VA_COUPLING_POS", "0.99")
        monkeypatch.setenv("ZERO_VA_COUPLING_NEG", "0.99")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-t5-persona-wins")
        expected_pos, expected_neg = big_five_to_va_coupling(e_val)
        assert driver.session.config.va_coupling_pos == pytest.approx(expected_pos), (
            f"persona 推导 va_coupling_pos 应优先于 env，期望 {expected_pos}，"
            f"实际 {driver.session.config.va_coupling_pos}"
        )
        assert driver.session.config.va_coupling_neg == pytest.approx(expected_neg), (
            f"persona 推导 va_coupling_neg 应优先于 env，期望 {expected_neg}，"
            f"实际 {driver.session.config.va_coupling_neg}"
        )
    finally:
        _os.unlink(tmp_path)


def test_build_chat_driver_env_va_coupling_used_when_no_persona_big_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """persona 无 big_five（且无显式 va_coupling）时，回落 env ZERO_VA_COUPLING_POS/NEG。"""
    import json
    import os as _os
    import tempfile

    import pytest

    from src.orchestration.chat_driver import build_chat_driver

    data = {"name": "no-big-five", "card": "test"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        tmp_path = f.name
    try:
        monkeypatch.setenv("ZERO_PERSONA_FILE", tmp_path)
        monkeypatch.setenv("ZERO_VA_COUPLING_POS", "0.45")
        monkeypatch.setenv("ZERO_VA_COUPLING_NEG", "0.72")
        monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
        driver = build_chat_driver(thread="test-t5-env-fallback")
        assert driver.session.config.va_coupling_pos == pytest.approx(0.45), (
            f"无 persona va_coupling 时应回落 env，实际 {driver.session.config.va_coupling_pos}"
        )
        assert driver.session.config.va_coupling_neg == pytest.approx(0.72), (
            f"无 persona va_coupling 时应回落 env，实际 {driver.session.config.va_coupling_neg}"
        )
    finally:
        _os.unlink(tmp_path)
