"""WARN-3b salience 门量纲修正回归（科学家议会 2026-07-22·Hill 归一 + threshold 0.25）。

验证 precision → normalize_precision(p,30) × 0.5 → SleepConsolidation(0.25) 门控端到端 4 边界：
- 量纲修正前 salience=precision×0.5（precision 无界·实测 28–72）→ 14–36，使门对全部正常 episode
  永真、退化为「precision 字段存在性检测」，升迁实际只靠 consolidation_count 单准则（失真）。
- 修正后归一到 (0, 0.5]，门恢复「precision≳30（实测中线）才升迁」的显著度梯度（双准则忠实）。

裁定落库：notes/2026-07-22-consolidation-salience-scale-council.md。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.memory.consolidation import SleepConsolidation
from src.memory.utils import normalize_precision

# 议会裁定的 4 边界（与 consolidation.py salience 计算处注释表一致）：
#   precision → normalize_precision(p,30) → ×0.5 → salience
_BOUNDARY = [
    (0.5, 0.016, 0.008),  # fallback（缺 precision 字段）：极低
    (28.0, 0.483, 0.241),  # 实测低端：略低于门 0.25
    (30.0, 0.500, 0.250),  # 半饱和中线：临界（= 门 0.25）
    (72.0, 0.706, 0.353),  # 实测高端：过门
]


@pytest.mark.parametrize(("precision", "exp_norm", "exp_salience"), _BOUNDARY)
def test_normalize_precision_boundary(
    precision: float, exp_norm: float, exp_salience: float
) -> None:
    """normalize_precision(p,30) 与 ×0.5 后 salience 落在议会裁定量级（±0.005）。"""
    norm = normalize_precision(precision, scale=30.0)
    assert norm == pytest.approx(exp_norm, abs=0.005)
    assert norm * 0.5 == pytest.approx(exp_salience, abs=0.005)


def test_normalize_precision_scale_nonpositive_returns_zero() -> None:
    """scale<=0 → 返回 0.0（防除零·utils guard）。"""
    assert normalize_precision(50.0, scale=0.0) == 0.0
    assert normalize_precision(50.0, scale=-1.0) == 0.0


def test_normalize_precision_monotone() -> None:
    """Hill 归一严格单调（数学席：单调变换保序·不改 precision 相对排序）。"""
    vals = [normalize_precision(p, 30.0) for p in (0.5, 28.0, 30.0, 72.0)]
    assert vals == sorted(vals)
    assert len(set(vals)) == 4  # 严格单调·无并列


@pytest.mark.parametrize(
    ("precision", "should_migrate"),
    [(0.5, False), (28.0, False), (30.0, True), (72.0, True)],
)
def test_salience_gate_after_scale_fix(precision: float, should_migrate: bool) -> None:
    """端到端：precision → salience → SleepConsolidation(0.25, cc_min=3) 门控。

    量纲修正后门恢复显著度梯度：consolidation_count 满足时，precision<30 不升迁、>=30 升迁。
    这正是修正前（salience=precision×0.5=14~36 恒 > 门）无法区分的显著度梯度。
    """
    salience = normalize_precision(precision, scale=30.0) * 0.5
    ep = {
        "episode_id": f"ep-{precision}",
        "scope": "session",
        "salience": salience,
        "consolidation_count": 3,  # cc 准则满足 → 单独考察 salience 门
    }
    _, ids = SleepConsolidation(salience_threshold=0.25, consolidation_count_min=3).compute(
        [ep], now=datetime.now(UTC)
    )
    assert (f"ep-{precision}" in ids) is should_migrate


def test_scale_fix_prevents_existence_check_degeneration() -> None:
    """回归护栏：修正后「缺 precision 字段（fallback 0.5）」不再因量纲失真而被特殊对待。

    修正前 fallback precision=0.5 → salience=0.25 < 门 0.3（被挡）、正常 28–72 → 14–36（永过），
    门退化为「有无 precision 字段」的存在性检测。修正后 fallback → salience≈0.008，
    与低 precision（28→0.241）同处「不升迁」侧，门语义统一为显著度梯度而非字段存在性。
    """
    fallback_salience = normalize_precision(0.5, scale=30.0) * 0.5
    low_salience = normalize_precision(28.0, scale=30.0) * 0.5
    high_salience = normalize_precision(72.0, scale=30.0) * 0.5
    # fallback 与低 precision 同侧（都不升迁）、高 precision 才升迁 —— 显著度梯度而非存在性
    assert fallback_salience < 0.25
    assert low_salience < 0.25
    assert high_salience >= 0.25
    assert fallback_salience < low_salience < high_salience  # 单调梯度
