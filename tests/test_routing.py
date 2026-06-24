"""T6.2 条件边路由函数独立单测：mood 后路由 + 语言回路路由。（G2）"""

from __future__ import annotations

from src.orchestration.graph import route_after_language, route_after_mood
from src.orchestration.state import AffectState


def test_route_to_regulation_when_enabled() -> None:
    assert route_after_mood(AffectState(regulation_enabled=True)) == "regulation"


def test_route_to_expression_when_disabled() -> None:
    assert route_after_mood(AffectState(regulation_enabled=False)) == "expression"


def test_route_to_language_takes_priority() -> None:
    # language 开启时优先进语言回路（高于 regulation）
    state = AffectState(language_enabled=True, regulation_enabled=True)
    assert route_after_mood(state) == "language"


def test_language_loop_repeats_when_inconsistent() -> None:
    # 不一致（dist>τ）且未达上限 → 回 language 重写
    state = AffectState(language_consistency=0.9, language_iter=1, language_max_iters=3)
    assert route_after_language(state) == "language"


def test_language_loop_exits_on_convergence() -> None:
    # 收敛（dist<=τ）→ 进下游 expression
    state = AffectState(language_consistency=0.05, language_iter=1)
    assert route_after_language(state) == "expression"


def test_language_loop_exits_on_max_iters() -> None:
    # 达终止上限即便仍不一致也退出；regulation 开启时进 regulation
    state = AffectState(
        language_consistency=0.9,
        language_iter=3,
        language_max_iters=3,
        regulation_enabled=True,
    )
    assert route_after_language(state) == "regulation"
