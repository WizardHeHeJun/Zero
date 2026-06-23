"""T6.2 条件边路由函数独立单测。（G2）"""

from __future__ import annotations

from src.orchestration.graph import route_after_affect_core
from src.orchestration.state import AffectState


def test_route_to_regulation_when_enabled() -> None:
    assert route_after_affect_core(AffectState(regulation_enabled=True)) == "regulation"


def test_route_to_expression_when_disabled() -> None:
    assert route_after_affect_core(AffectState(regulation_enabled=False)) == "expression"
