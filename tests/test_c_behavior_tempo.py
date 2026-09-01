"""c-behavior-tempo：C→行为节奏偏置 v1（2026-08-31 设计门·PRP/c-behavior-tempo/design.md）。

T1 审慎措辞风格句（三档 ±0.3 + 预设文案闭集抑制）+ T2 解码温度偏置（连续**无死区**、
converse 专用叠加，`self.temperature`/`_compose` 不动）。门 ZERO_PERSONA_C_STYLE /
ZERO_PERSONA_C_TEMPO 默认关（两门独立，CS 席裁定）。

KNOWN_MISS（诚实边界，勿当 bug「顺手修」）：闭集判据是**精确相等**匹配——对预设卡文案
做小改动后仍保留大部分风格词的卡不会被抑制（test_preset_text_mutation_escapes 钉住该
方向）。逃逸 = 回退为同向强化注入，自愈式降级（议会张力①已裁），把判据改成子串/词表
匹配反而落入 text-predicate-admission 要挡的非锚定匹配形态。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.language_openai import OpenAILanguageModel
from src.agents.persona import (
    K_C_TEMPO,
    PRESET_CARD_TEXTS,
    Persona,
    _persona_from_dict,
    conscientiousness_to_style_hint,
    conscientiousness_to_temperature_offset,
)
from src.orchestration.chat_driver import (
    persona_converse_temperature_offset,
    persona_prompt_text,
)

_PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


def _load_preset(name: str) -> dict[str, Any]:
    with open(_PERSONAS_DIR / f"{name}.example.json", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


# ── T1 查表纯函数 ────────────────────────────────────────────────────────────


def test_high_c_custom_card_injects_deliberate() -> None:
    """自定义卡（非预设文案）+ 高 C：注入审慎档。"""
    hint = conscientiousness_to_style_hint(0.6, "一张自定义人设卡。")
    assert "先理一遍思路" in hint


def test_low_c_injects_spontaneous() -> None:
    hint = conscientiousness_to_style_hint(-0.6, "一张自定义人设卡。")
    assert "想到什么说什么" in hint


def test_deadzone_silent() -> None:
    assert conscientiousness_to_style_hint(0.29, "x") == ""
    assert conscientiousness_to_style_hint(-0.29, "x") == ""


def test_boundary_inclusive() -> None:
    """阈值 ≥/≤ 含边界（±0.3 恰好命中）。"""
    assert conscientiousness_to_style_hint(0.3, "x") != ""
    assert conscientiousness_to_style_hint(-0.3, "x") != ""


def test_no_typology_words_in_hints() -> None:
    """议会措辞纪律：注入文本不得出现类型学词（「尽责」「冲动」）。"""
    for c in (0.6, -0.6):
        hint = conscientiousness_to_style_hint(c, "x")
        assert "尽责" not in hint and "冲动" not in hint


def test_preset_card_text_suppressed_even_with_high_c() -> None:
    """双源抑制：orderly-planner 的 card 逐字文本（规则 5 治理，已含 C 风格词）→ 不注入。"""
    card = str(_load_preset("orderly-planner")["card"])
    assert conscientiousness_to_style_hint(0.6, card) == ""


def test_preset_text_mutation_escapes() -> None:
    """KNOWN_MISS 方向成文：对预设文案做一字改动即逃逸闭集 → 回退为同向强化注入。

    这是判据的**已登记限制**（自愈式降级，fail-safe），不是待修 bug——本用例红了说明
    有人把精确匹配改成了语义近似匹配，先回 design.md 张力①复核再动。
    """
    card = str(_load_preset("orderly-planner")["card"]) + "改"
    assert conscientiousness_to_style_hint(0.6, card) != ""


def test_preset_sync_guard() -> None:
    """同步守卫（机械化 PR checklist）：所有带 big_five 的 *.example.json（=规则 5 治理卡）
    card 必须逐字收录进 PRESET_CARD_TEXTS。红 = 新增/编辑了预设卡但没同步闭集。"""
    governed = 0
    for path in sorted(_PERSONAS_DIR.glob("*.example.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "big_five" not in data:
            continue  # persona.example.json 等非治理卡：无 big_five，不受规则 5 约束
        governed += 1
        assert str(data["card"]) in PRESET_CARD_TEXTS, path.name
    assert governed >= 6, "预设库治理卡应 ≥6 张——否则本守卫没在测东西"


def test_preset_library_t1_all_suppressed() -> None:
    """分布锚点（对照 B1 的 0/6 占比表先例）：当前预设库 T1 注入 0/N——C 主导卡
    （orderly-planner C=0.6、steady-companion C=0.8）被闭集抑制，其余落 |C|<0.3 死区。
    红 = 加了 C 主导新卡且未收录闭集（先补 PRESET_CARD_TEXTS，同步守卫同时会红）。"""
    for path in sorted(_PERSONAS_DIR.glob("*.example.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert _persona_from_dict(data).c_style_hint == "", path.name


# ── T0 前置修复：big_five 输入 clamp ─────────────────────────────────────────


def test_big_five_out_of_range_clamped() -> None:
    """越界输入 clamp 到 [-1,1]（议会前置修复）：c=5.0 按 1.0 派生，偏移不越 -K。"""
    p = _persona_from_dict({"card": "x", "big_five": {"conscientiousness": 5.0}})
    assert p.conscientiousness == 1.0
    assert p.c_tempo_offset == pytest.approx(-K_C_TEMPO)


def test_big_five_in_range_unchanged() -> None:
    """[-1,1] 内取值逐字不变（clamp 零回归面）。"""
    p = _persona_from_dict({"card": "x", "big_five": {"conscientiousness": 0.6}})
    assert p.conscientiousness == pytest.approx(0.6)
    assert p.c_tempo_offset == pytest.approx(-0.15)


# ── T2 查表纯函数 + 不变式 ───────────────────────────────────────────────────


def test_offset_table() -> None:
    """连续映射查表（K=0.25 改坏必红）：无死区、对称双向、越界 clamp。"""
    assert conscientiousness_to_temperature_offset(0.0) == 0.0
    assert conscientiousness_to_temperature_offset(0.3) == pytest.approx(-0.075)
    assert conscientiousness_to_temperature_offset(-0.3) == pytest.approx(0.075)
    assert conscientiousness_to_temperature_offset(1.0) == pytest.approx(-0.25)
    assert conscientiousness_to_temperature_offset(-1.0) == pytest.approx(0.25)
    assert conscientiousness_to_temperature_offset(5.0) == pytest.approx(-K_C_TEMPO)


def test_k_lower_bound_from_math_seat() -> None:
    """数学席可辨性下界：TV=K·|c|/0.25 在 c=0.6 预设档须 ≥0.5 ⇒ K≥0.208。K 调小必红。"""
    assert K_C_TEMPO * 0.6 / 0.25 >= 0.5


def test_truncation_unreachable_invariant() -> None:
    """截断不可达不变式：0.8 − K·1 − 0.1 > 0 ⇒ converse 的 max(0,·) 在 |c|≤1 域内不触发。
    K 调大到 0.7+ 必红——调 K 前先看 persona.K_C_TEMPO 注释与文本层占比表。"""
    assert 0.8 - K_C_TEMPO * 1.0 - 0.1 > 0.0


# ── Persona 派生 ────────────────────────────────────────────────────────────


def test_persona_without_big_five_c_fields_default() -> None:
    """无 big_five：三字段全默认——「没配置」与「配置成中性」语义分野。"""
    p = Persona()
    assert p.conscientiousness is None
    assert p.c_style_hint == ""
    assert p.c_tempo_offset is None


def test_persona_from_big_five_derives_c_fields() -> None:
    p = _persona_from_dict({"card": "自定义卡", "big_five": {"conscientiousness": 0.6}})
    assert p.conscientiousness == pytest.approx(0.6)
    assert p.c_style_hint != ""
    assert p.c_tempo_offset == pytest.approx(-0.15)


# ── T1 拼装端到端锚点 ────────────────────────────────────────────────────────

_CUSTOM_HIGH_C = _persona_from_dict(
    {"card": "一张自定义人设卡。", "big_five": {"conscientiousness": 0.6}}
)


def test_c_gate_off_prompt_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """门关（默认）：即便 hint 非空，注入文本恰为 card 字节原文——端到端零回归。"""
    monkeypatch.delenv("ZERO_PERSONA_C_STYLE", raising=False)
    monkeypatch.delenv("ZERO_PERSONA_DOMINANCE_STYLE", raising=False)
    assert persona_prompt_text(_CUSTOM_HIGH_C) == _CUSTOM_HIGH_C.card


def test_c_gate_on_hint_appended_not_written_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_PERSONA_C_STYLE", "true")
    monkeypatch.delenv("ZERO_PERSONA_DOMINANCE_STYLE", raising=False)
    text = persona_prompt_text(_CUSTOM_HIGH_C)
    assert text.startswith(_CUSTOM_HIGH_C.card) and "先理一遍思路" in text
    assert "先理一遍思路" not in _CUSTOM_HIGH_C.card  # 不写回 card


def test_c_gate_on_preset_card_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """门开但 hint 被闭集抑制（预设卡）：返回值仍为 card 字节原文。"""
    monkeypatch.setenv("ZERO_PERSONA_C_STYLE", "true")
    monkeypatch.delenv("ZERO_PERSONA_DOMINANCE_STYLE", raising=False)
    p = _persona_from_dict(_load_preset("orderly-planner"))
    assert persona_prompt_text(p) == p.card


def test_b1_and_t1_both_on_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """组合矩阵「B1开×T1开」（CS 席点名高风险格）：同一字符串场两句共存，顺序 card→D→C。"""
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    monkeypatch.setenv("ZERO_PERSONA_C_STYLE", "true")
    p = _persona_from_dict(
        {"card": "自定义卡。", "big_five": {"openness": 1.0, "conscientiousness": 1.0}}
    )
    text = persona_prompt_text(p)
    assert text.startswith("自定义卡。")
    assert "直截了当" in text and "先理一遍思路" in text
    assert text.index("直截了当") < text.index("先理一遍思路")


def test_b1_gate_alone_unchanged_by_t1_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 单独开（T1 关）：返回值与 B1 落地时的既有行为逐字一致（card\\nD句）。"""
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    monkeypatch.delenv("ZERO_PERSONA_C_STYLE", raising=False)
    p = _persona_from_dict(
        {"card": "自定义卡。", "big_five": {"openness": 1.0, "conscientiousness": 1.0}}
    )
    assert persona_prompt_text(p) == f"{p.card}\n{p.dominance_style_hint}"


# ── T2 装配门控 ─────────────────────────────────────────────────────────────


def test_tempo_helper_gate_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_PERSONA_C_TEMPO", raising=False)
    assert persona_converse_temperature_offset(_CUSTOM_HIGH_C) is None


def test_tempo_helper_gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_PERSONA_C_TEMPO", "true")
    assert persona_converse_temperature_offset(_CUSTOM_HIGH_C) == pytest.approx(-0.15)
    assert persona_converse_temperature_offset(Persona()) is None  # 无 big_five
    p0 = _persona_from_dict({"card": "x", "big_five": {"conscientiousness": 0.0}})
    assert persona_converse_temperature_offset(p0) is None  # 连续映射零点：不传参


def test_t2_independent_of_b1_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """组合矩阵「B1开×T2开」：偏移值与 B1 门状态无关（字符串通道 ⟂ 数值通道）。"""
    monkeypatch.setenv("ZERO_PERSONA_C_TEMPO", "true")
    monkeypatch.delenv("ZERO_PERSONA_DOMINANCE_STYLE", raising=False)
    off_b1_off = persona_converse_temperature_offset(_CUSTOM_HIGH_C)
    monkeypatch.setenv("ZERO_PERSONA_DOMINANCE_STYLE", "true")
    assert persona_converse_temperature_offset(_CUSTOM_HIGH_C) == off_b1_off


# ── T2 adapter 层：converse 专用叠加 + _compose 隔离 ─────────────────────────


class _CapturingClient:
    """捕获 chat.completions.create 实参（监视生产调用点，不发网络请求）。"""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.captured = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="0.0 0.0"))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


_HISTORY = [{"role": "user", "content": "在吗"}]


async def test_default_offset_converse_formula_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """零回归锚点：默认 offset=0.0 时 converse 温度 = 0.8 + 抖动，与改前公式逐字等价。"""
    import src.agents.language_openai as lo

    monkeypatch.delenv("ZERO_FACTUAL_MODE", raising=False)
    monkeypatch.setattr(lo.random, "uniform", lambda a, b: 0.12345)
    client = _CapturingClient()
    lm = OpenAILanguageModel(model="test-model", client=client)
    assert lm.converse_temperature_offset == 0.0
    await lm.converse(_HISTORY, (0.0, 0.0))
    assert client.captured["temperature"] == pytest.approx(0.8 + 0.12345)


async def test_offset_applies_to_converse_not_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_compose` 隔离（数学席同构站点裁定）：偏移只进 converse，`_compose` 仍用
    `self.temperature`——断言取自生产调用点实际收到的 kwargs，不自己重算期望。"""
    import src.agents.language_openai as lo

    monkeypatch.delenv("ZERO_FACTUAL_MODE", raising=False)
    monkeypatch.setattr(lo.random, "uniform", lambda a, b: 0.0)
    client = _CapturingClient()
    lm = OpenAILanguageModel(model="test-model", client=client, converse_temperature_offset=-0.15)
    await lm.converse(_HISTORY, (0.0, 0.0))
    assert client.captured["temperature"] == pytest.approx(0.8 - 0.15)
    await lm._compose((0.0, 0.0), "上下文", "", None)
    assert client.captured["temperature"] == 0.8  # 字面等：offset 未泄漏进研究路径


# ── 工厂端到端：kwarg 省略式零回归 ───────────────────────────────────────────


def _spy_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, thread: str) -> dict[str, Any]:
    """monkeypatch 掉延迟 import 的 OpenAILanguageModel，捕获 build_chat_driver 传的构造实参。"""
    import src.agents.language_openai as lo
    from src.orchestration.chat_driver import build_chat_driver

    calls: dict[str, Any] = {}

    class _SpyLM:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    monkeypatch.setattr(lo, "OpenAILanguageModel", _SpyLM)
    persona_file = tmp_path / "p.json"
    persona_file.write_text(
        json.dumps({"card": "自定义高C卡", "big_five": {"conscientiousness": 0.6}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(persona_file))
    monkeypatch.setenv("ZERO_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ZERO_OPENAI_MODEL", "test-model")
    build_chat_driver(thread=thread)
    return calls


def test_build_chat_driver_gate_off_omits_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """门未设 → 工厂**省略** kwarg（非传 0.0）——唯一真源留 adapter 默认值。"""
    monkeypatch.delenv("ZERO_PERSONA_C_TEMPO", raising=False)
    calls = _spy_build(monkeypatch, tmp_path, "c-tempo-spy-off")
    assert calls, "spy 未捕获到构造调用——本用例没在测东西"
    assert "converse_temperature_offset" not in calls


def test_build_chat_driver_gate_on_passes_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ZERO_PERSONA_C_TEMPO", "true")
    calls = _spy_build(monkeypatch, tmp_path, "c-tempo-spy-on")
    assert calls["converse_temperature_offset"] == pytest.approx(-0.15)
