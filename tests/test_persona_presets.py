"""预设人格库机械层守卫（议会 2026-08-31 设计门）。

裁决纪要：notes/2026-08-31-persona-presets-council.md。

三类守卫（CS 席规格）：glob 驱动可加载 / big_five 数值域 / 卡间 setpoint 区分度。
glob 而非硬编码文件名——新增/删除预设卡不用改测试（防测试列表漂移）。

区分度锚点（数学席 + 修正案）：非豁免 pairwise ‖Δsetpoint‖₂ ≥ EPS=0.15
（chat_driver 直混权重 0.4 × 噪声地板 0.05 + 边际反解），浮点用 EPS−1e-9 容差
——高O 卡压线 0.15 等号过是 O 维系数（0.15）的真实上限，勿调 EPS 避开。
豁免规则（moderator 裁定，「结构性零」语义勿放宽）：当且仅当一对卡的 big_five 差异
完全落在 P/A' 方程系数均为 0 的维度（目前仅 conscientiousness）时豁免分布层锚点，
豁免须显式跳过并打印原因，不得静默；其方向层证据由 card 文案的尽责性锚点词承担。
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from src.agents.persona import _persona_from_dict, load_persona

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"
PRESET_PATHS = sorted(PERSONAS_DIR.glob("*.example.json"))

EPS = 0.15
_FLOAT_TOL = 1e-9
# P/A' 方程系数均为 0 的维度（big_five_to_pad：C 只进未被 setpoint 消费的 Dominance）
_STRUCTURAL_ZERO_DIMS = {"conscientiousness"}
_BIG5_KEYS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict), f"{path.name}: 顶层须为 JSON 对象"
    return data


def test_presets_exist() -> None:
    """库至少含中性卡 + 议会定案的 6 张预设卡（glob 空转 = 目录被挪，守卫失去对象）。"""
    assert len(PRESET_PATHS) >= 7, f"personas/*.example.json 仅 {len(PRESET_PATHS)} 份"


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda p: p.stem)
def test_preset_loads_without_error(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """每张卡经 load_persona 全链加载不抛错，且至少有 L1 文本。"""
    monkeypatch.setenv("ZERO_PERSONA_FILE", str(path))
    persona = load_persona()
    assert persona.card, f"{path.name}: card 文本为空"


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda p: p.stem)
def test_preset_big_five_domain(path: Path) -> None:
    """big_five 各维 ∈ [-1,1]（防误填心理量表原始分等错量纲——数据质量门在测试侧，
    产品侧 clamp 容错刻意保留）。"""
    data = _load_raw(path)
    if "big_five" not in data:
        return  # 中性卡无 big_five，合法
    for key in _BIG5_KEYS:
        value = float(data["big_five"].get(key, 0.0))
        assert -1.0 <= value <= 1.0, f"{path.name}: big_five.{key}={value} 超出 [-1,1]"


def test_preset_pairwise_setpoint_distinct() -> None:
    """分布层锚点：非豁免 pairwise ‖Δsetpoint‖₂ ≥ EPS（含浮点容差）。

    豁免对显式打印跳过原因（议会必改 6：登记不静默）。
    """
    cards: dict[str, tuple[tuple[float, float], dict[str, float]]] = {}
    for path in PRESET_PATHS:
        data = _load_raw(path)
        big5 = {k: float(data.get("big_five", {}).get(k, 0.0)) for k in _BIG5_KEYS}
        cards[path.stem] = (_persona_from_dict(data).setpoint, big5)

    failures: list[str] = []
    for (name_a, (sp_a, b5_a)), (name_b, (sp_b, b5_b)) in itertools.combinations(cards.items(), 2):
        distance = math.dist(sp_a, sp_b)
        if distance >= EPS - _FLOAT_TOL:
            continue
        differing = {k for k in _BIG5_KEYS if abs(b5_a[k] - b5_b[k]) > _FLOAT_TOL}
        if differing and differing <= _STRUCTURAL_ZERO_DIMS:
            print(
                f"[豁免登记] {name_a} ↔ {name_b}: 差异仅在结构性零维 {sorted(differing)}"
                f"（P/A' 系数均 0），分布层锚点豁免，方向层由 card 文案锚点词承担"
            )
            continue
        failures.append(f"{name_a} ↔ {name_b}: ‖Δsetpoint‖₂={distance:.3f} < {EPS}")
    assert not failures, "卡间区分度不达标：\n" + "\n".join(failures)
