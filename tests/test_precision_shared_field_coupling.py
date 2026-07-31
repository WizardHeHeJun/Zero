"""`precision=` 共享字段「一钱多用」的数值回归锁（议会 2026-07-31 D3 + D4）。

`IDENTITY_MEMORY_PRECISION` / `SEED_MEMORY_PRECISION` 这两个常量把 episode 的
`precision=` 字段顶到 40.0，本意只是越过召回注入门；但同一字段还被
`memory_recall._rank_episodes` 的 importance 维与 `consolidation.EbbinghausDecay`
的 `a_eff` 消费——**改这两个常量会同时改变排序与遗忘行为**。

这两条锁把该耦合的量级钉住：改常量（或改 `ZERO_RECALL_IMPORTANCE_SCALE`）时必须变红，
而不是静默改变身份/种子记忆的排序与遗忘。本仓教训见
`ai-docs/pitfalls.md`「affect_precision 是无界方差倒数」系列与「绿灯必须先证明能红」。

⚠ 全部数值由**生产函数**算出（`normalized_importance` / `normalize_precision` /
`EbbinghausDecay`），不在测试里另写一份公式——否则断言的是测试代码自己。
"""

from __future__ import annotations

import pytest

from src.memory.consolidation import EbbinghausDecay
from src.memory.utils import normalize_precision
from src.orchestration.chat_driver import SEED_MEMORY_PRECISION
from src.orchestration.memory_recall import normalized_importance
from src.orchestration.supervisor import IDENTITY_MEMORY_PRECISION

# 身份轮实测的真实 affect_precision（100 轮实跑，未被下限顶高时的值）
RAW_PRECISION = 8.5633
IMPORTANCE_SCALE = 30.0  # ZERO_RECALL_IMPORTANCE_SCALE 默认
GAMMA = 0.33  # _rank_episodes 的 importance 维权重默认

EXPECTED_DELTA_SCORE = 0.1153  # 排序侧固定加成
EXPECTED_AEFF_RATIO = 1.604  # 遗忘侧振幅比值
TOL = 1e-3


def _delta_score(lifted: float) -> float:
    """覆写 precision 给排序分带来的固定加成（走生产的 normalized_importance）。"""
    raw = normalized_importance(f"precision={RAW_PRECISION:.2f}", IMPORTANCE_SCALE)
    high = normalized_importance(f"precision={lifted:.2f}", IMPORTANCE_SCALE)
    return GAMMA * (high - raw)


def _aeff_ratio(lifted: float) -> float:
    """覆写 precision 给遗忘振幅带来的比值。

    走生产的 `normalize_precision` 与 `EbbinghausDecay` 的参数，不另写公式。
    """
    decay = EbbinghausDecay()
    sal_raw = normalize_precision(RAW_PRECISION, IMPORTANCE_SCALE) * 0.5
    sal_high = normalize_precision(lifted, IMPORTANCE_SCALE) * 0.5
    a_raw = decay.a * sal_raw**decay.kappa
    a_high = decay.a * sal_high**decay.kappa
    return a_high / a_raw


@pytest.mark.parametrize(
    "constant", [IDENTITY_MEMORY_PRECISION, SEED_MEMORY_PRECISION], ids=["identity", "seed"]
)
def test_precision_floor_ranking_side_effect_is_locked(constant: float) -> None:
    """排序侧：覆写带来 +0.1153 的固定加成（占三维总权重 11.5%）。

    D4：两个常量同构、同耦合，先例不免检，故一并锁。
    """
    assert _delta_score(constant) == pytest.approx(EXPECTED_DELTA_SCORE, abs=TOL), (
        "precision 下限对召回排序的加成变了。若这是有意改动，请同步更新常量旁的披露注释"
        "与本锁的期望值，并确认遗忘侧（下一条锁）的影响也被重新评估。"
    )


@pytest.mark.parametrize(
    "constant", [IDENTITY_MEMORY_PRECISION, SEED_MEMORY_PRECISION], ids=["identity", "seed"]
)
def test_precision_floor_forgetting_side_effect_is_locked(constant: float) -> None:
    """遗忘侧：覆写使 a_eff 变为 1.604 倍，且这是**振幅项**、不随 Δt 衰减。"""
    assert _aeff_ratio(constant) == pytest.approx(EXPECTED_AEFF_RATIO, abs=TOL), (
        "precision 下限对遗忘振幅的调制变了。注意该倍率不随时间衰减，是终身乘性优势。"
    )


def test_two_constants_are_identical() -> None:
    """D4 的前提：两个常量同值同构。若将来分叉，上面的参数化锁需拆开重新标定。"""
    assert IDENTITY_MEMORY_PRECISION == SEED_MEMORY_PRECISION


@pytest.mark.parametrize("wrong", [20.0, 30.0, 60.0, 100.0])
def test_locks_go_red_when_constant_changes(wrong: float) -> None:
    """**证明这两条锁能红**（议会 D3 验收要求，也是本仓「绿灯必须先证明能红」纪律）。

    换任何别的常量值，两条锁的期望值都不再成立——说明它们确实盯着这个常量，
    而不是碰巧写了两个恒成立的数。
    """
    assert abs(_delta_score(wrong) - EXPECTED_DELTA_SCORE) > TOL, (
        f"precision={wrong} 时排序加成仍等于期望值，这条锁对常量变化不敏感"
    )
    assert abs(_aeff_ratio(wrong) - EXPECTED_AEFF_RATIO) > TOL, (
        f"precision={wrong} 时遗忘比值仍等于期望值，这条锁对常量变化不敏感"
    )


def test_scale_change_also_breaks_the_lock() -> None:
    """`ZERO_RECALL_IMPORTANCE_SCALE` 变了同样应变红——它与常量共同决定这两个量级。"""
    raw = normalized_importance(f"precision={RAW_PRECISION:.2f}", 60.0)
    high = normalized_importance(f"precision={IDENTITY_MEMORY_PRECISION:.2f}", 60.0)
    assert abs(GAMMA * (high - raw) - EXPECTED_DELTA_SCORE) > TOL
