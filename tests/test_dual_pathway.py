"""T6.4 双通路：Regulation 恒等→两头一致；开启→两头差异；并校验 4 通道结构化。（G3, G5）"""

from __future__ import annotations

from typing import Any

from src.agents.expression import ExpressionAgent
from src.agents.regulation import RegulationAgent
from src.orchestration.state import AffectState

CHANNELS = {"facs_au", "text_label", "physiology", "prosody"}


def express(regulation_enabled: bool) -> dict[str, Any]:
    """走 Regulation → Expression，返回 expression 字段。"""
    state = AffectState(affect_sample=(0.8, 0.7), regulation_enabled=regulation_enabled)
    state = state.model_copy(update=RegulationAgent()(state))
    out = ExpressionAgent()(state)
    return out["expression"]


def test_two_heads_identical_when_regulation_disabled() -> None:
    expr = express(regulation_enabled=False)
    assert expr["spontaneous"] == expr["voluntary"]


def test_two_heads_differ_when_regulation_enabled() -> None:
    expr = express(regulation_enabled=True)
    assert expr["spontaneous"] != expr["voluntary"]


def test_each_head_emits_four_channels_and_va() -> None:
    expr = express(regulation_enabled=True)
    assert expr["valence_arousal"] is not None
    for head in ("spontaneous", "voluntary"):
        assert CHANNELS.issubset(expr[head]), head
