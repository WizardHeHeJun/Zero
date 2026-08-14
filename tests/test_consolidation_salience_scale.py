"""WARN-3b salience 计算公式的边界值回归（科学家议会 2026-07-22·Hill 归一 + threshold 0.25）。

验证 precision → normalize_precision(p,30) × 0.5 → salience 计算公式端到端 4 边界（历史门值
0.25 为参照，仅用于标注边界位置；门控对象 SleepConsolidation 已于 2026-08-13 复裁退役
[PRP/sleep-consolidation-verdict/design.md]，本文件不再验证任何门控行为，只验证
normalize_precision 本身的边界值/单调性/防护逻辑）：
- 量纲修正前 salience=precision×0.5（precision 无界·实测 28–72）→ 14–36，会使假设中的门对
  全部正常 episode 永真、退化为「precision 字段存在性检测」（历史失真，已随门控退役失去实指）。
- 修正后归一到 (0, 0.5]，恢复「precision≳30（实测中线）」附近的显著度梯度。

裁定落库：notes/2026-07-22-consolidation-salience-scale-council.md；
门控退役裁定：PRP/sleep-consolidation-verdict/design.md（2026-08-13）。
"""

from __future__ import annotations

import pytest

from src.memory.utils import normalize_precision

# 历史门值 0.25 为参照边界位置（不代表任何仍在生效的门控）：
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


def test_scale_fix_prevents_existence_check_degeneration() -> None:
    """回归护栏：修正后「缺 precision 字段（fallback 0.5）」不再因量纲失真而被特殊对待。

    历史门值 0.25 仅作参照边界：修正前 fallback precision=0.5 → salience=0.25 恰在参照门
    附近、正常 28–72 → 14–36（远超参照门），若曾有门控则会退化为「有无 precision 字段」
    的存在性检测。修正后 fallback → salience≈0.008，与低 precision（28→0.241）同处参照门
    以下一侧，salience 计算公式呈现的是显著度梯度而非字段存在性。
    """
    fallback_salience = normalize_precision(0.5, scale=30.0) * 0.5
    low_salience = normalize_precision(28.0, scale=30.0) * 0.5
    high_salience = normalize_precision(72.0, scale=30.0) * 0.5
    # fallback 与低 precision 同侧（均低于参照门）、高 precision 越过参照门 —— 显著度梯度
    assert fallback_salience < 0.25
    assert low_salience < 0.25
    assert high_salience >= 0.25
    assert fallback_salience < low_salience < high_salience  # 单调梯度
