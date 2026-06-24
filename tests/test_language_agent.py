"""LanguageAgent + affect_math 语言工具单测：门控、契约、双向互调、一致性。"""

from __future__ import annotations

import pytest

from src.agents.affect_math import affect_distance, reconcile_affect
from src.agents.language import LanguageAgent
from src.orchestration.state import AffectState, Stimulus

FIELDS = set(AffectState.model_fields)


async def test_noop_when_disabled() -> None:
    out = await LanguageAgent()(AffectState(affect_sample=(0.4, 0.3), language_enabled=False))
    assert out == {}


async def test_noop_without_sample() -> None:
    assert await LanguageAgent()(AffectState(language_enabled=True)) == {}


async def test_placeholder_generates_text_and_consistent_affect() -> None:
    state = AffectState(
        stimulus=Stimulus(name="gift"),
        affect_sample=(0.6, 0.4),
        language_enabled=True,
    )
    before = state.model_copy(deep=True)
    out = await LanguageAgent()(state)
    assert isinstance(out["language_text"], str) and out["language_text"]
    assert out["language_affect"] == (0.6, 0.4)  # 占位回传目标情感
    assert out["language_consistency"] == 0.0  # 一致 → 一次收敛
    assert out["language_iter"] == 1
    assert set(out).issubset(FIELDS)  # 只返回合法 state 字段
    assert state == before  # 未原地 mutate 入参


async def test_rewrite_reconciles_affect_toward_language() -> None:
    # 回路重写（iter>0）：先把内核 e* 向上一轮语言情感拉拢，写回 affect_sample 增量
    state = AffectState(
        stimulus=Stimulus(name="s"),
        affect_sample=(0.0, 0.0),
        language_affect=(0.8, 0.8),
        language_consistency=0.9,
        language_iter=1,
        language_enabled=True,
    )
    out = await LanguageAgent()(state)
    assert out["affect_sample"] == reconcile_affect((0.0, 0.0), (0.8, 0.8))


def test_affect_distance_and_reconcile_math() -> None:
    assert affect_distance((0.0, 0.0), (0.3, 0.4)) == pytest.approx(0.5)
    assert reconcile_affect((0.0, 0.0), (1.0, 1.0), weight=0.5) == (0.5, 0.5)
    assert reconcile_affect((0.0, 0.0), (1.0, 1.0), weight=0.0) == (0.0, 0.0)
