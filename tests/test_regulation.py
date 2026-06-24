"""RegulationAgent 策略单测：identity（关）/ suppression（默认）/ reappraisal（Gross）。"""

from __future__ import annotations

from src.agents.affect_math import reappraise
from src.agents.regulation import RegulationAgent
from src.orchestration.state import AffectState


def test_identity_when_disabled() -> None:
    out = RegulationAgent()(AffectState(affect_sample=(-0.8, 0.7), regulation_enabled=False))
    assert out["regulated_affect"] == (-0.8, 0.7)
    assert out["trace"][0]["strategy"] == "identity"


def test_suppression_is_default_strategy_and_unchanged() -> None:
    # 默认策略 = suppression，公式仍为 (0.3v, 0.5a)（零回归）
    out = RegulationAgent()(AffectState(affect_sample=(-0.8, 0.7), regulation_enabled=True))
    assert out["regulated_affect"] == (0.3 * -0.8, 0.5 * 0.7)
    assert out["trace"][0]["strategy"] == "suppression"


def test_reappraisal_strategy_uses_reappraise() -> None:
    state = AffectState(
        affect_sample=(-0.8, 0.7),
        regulation_enabled=True,
        regulation_strategy="reappraisal",
    )
    out = RegulationAgent()(state)
    assert out["regulated_affect"] == reappraise((-0.8, 0.7))
    assert out["trace"][0]["strategy"] == "reappraisal"


def test_reappraisal_less_negative_than_suppression() -> None:
    neg = (-0.8, 0.7)
    supp = RegulationAgent()(AffectState(affect_sample=neg, regulation_enabled=True))
    reapp = RegulationAgent()(
        AffectState(affect_sample=neg, regulation_enabled=True, regulation_strategy="reappraisal")
    )
    # 重评比抑制：效价更不负、唤醒更低（早期干预、改体验更有效）
    assert reapp["regulated_affect"][0] > supp["regulated_affect"][0]
    assert reapp["regulated_affect"][1] < supp["regulated_affect"][1]


def test_noop_without_sample() -> None:
    assert RegulationAgent()(AffectState(regulation_enabled=True)) == {}
