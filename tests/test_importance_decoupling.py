"""IDENTITY_MEMORY_PRECISION 停止向 ②③ 扩散的解耦断言（PRP importance-signal · T5）。

议会 2026-07-31 判定该覆写为 Stevens 1946 尺度类型误用：一个只对**阈值判定**安全的
覆写，扩散进了要求**基数尺度**的排序（②）与遗忘调制（③）公式。

⚠ 该覆写**不退场**（原 PRP 表述已作废）：它存在的唯一目的就是让身份 episode 过
① 注入门（`inject_min`），而用户拍板的 D-B 保留 ① 继续读 `precision=`——退场等于
身份记忆重新变成召不回。实质目标是**停止扩散**：②③ 不再读 `precision=`。

本文件用「把常量换成别的值，②③ 读数必须纹丝不动」来钉死这一点——比断言
「②③ 用了新信号」更强，因为后者可能在实现里仍残留一条 precision 旁路。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.memory.consolidation import EbbinghausDecay
from src.memory.types import Fact, Scope
from src.memory.utils import importance_signal, parse_importance_tags
from src.orchestration.memory_recall import _rank_episodes

NOW = datetime(2026, 8, 12, tzinfo=UTC)
RANK_KW = {"alpha": 0.33, "beta": 0.34, "gamma": 0.33, "importance_scale": 30.0}


def _identity_episode(precision: float) -> str:
    """身份 episode 的真实 content：precision 取值由调用方给（模拟覆写取不同常量）。"""
    return (
        f"你说：我叫林川 | 情绪=平静(0.05,0.05) | precision={precision:.2f}"
        f" | streams=[] | value=0.000 | identity=name"
    )


def _fact(content: str, *, days_ago: float) -> Fact:
    return Fact(
        content=content,
        scope=Scope.USER,
        valid_at=NOW - timedelta(days=days_ago),
        key="u1",
        sim=0.5,
    )


# 覆写常量的候选取值：现行 40.0、退场后的裸值 8.56、以及两个任意值。
# ②③ 的读数必须对**全部**取值相同。
PRECISION_VARIANTS = [8.56, 40.0, 72.0, 999.0]


def test_rank_score_is_independent_of_precision_when_tag_mode_on() -> None:
    """② 召回排序：门开时，identity episode 的排序表现不随 precision 取值变化。

    构造一个「同分对手」：若 identity 条目的 importance 维仍受 precision 影响，
    不同 precision 下它与对手的相对顺序会翻转。
    """
    rival = _fact(
        "你说：随便聊聊 | 情绪=平静(0.10,0.10) | precision=45.00 | streams=[] | value=0.000",
        days_ago=1,
    )
    orders: set[tuple[str, ...]] = set()
    for p in PRECISION_VARIANTS:
        target = _fact(_identity_episode(p), days_ago=1)
        ranked = _rank_episodes([target, rival], NOW, tag_importance_enabled=True, **RANK_KW)
        orders.add(tuple("identity" if "identity=name" in f.content else "rival" for f in ranked))
    assert len(orders) == 1, f"② 排序仍受 precision 影响：出现了不同顺序 {orders}"


def test_rank_importance_value_is_independent_of_precision() -> None:
    """② 的 importance 维取值本身：门开时对 precision 完全不敏感（直接断言信号值）。"""
    values = {
        importance_signal(parse_importance_tags(_identity_episode(p))) for p in PRECISION_VARIANTS
    }
    assert len(values) == 1, f"tag 信号随 precision 变化：{values}"
    # identity 单 tag 命中 ⇒ 1.2×b0
    assert values.pop() == pytest.approx(0.6)


def test_decay_weight_is_independent_of_precision() -> None:
    """③ 遗忘调制：decay_weight 不随 precision 取值变化。"""
    weights: set[float] = set()
    for p in PRECISION_VARIANTS:
        content = _identity_episode(p)
        ep = {
            "episode_id": "e1",
            "scope": "user",
            "key": "u1",
            "content": content,
            "valid_at": NOW - timedelta(days=5),
            "importance": importance_signal(parse_importance_tags(content)),
        }
        (dw, _eid), *_ = EbbinghausDecay(kappa=1.0).compute([ep], now=NOW)[0]  # type: ignore[misc]
        weights.add(round(dw, 12))
    assert len(weights) == 1, f"③ decay_weight 仍受 precision 影响：{weights}"


def test_gate_off_still_reads_precision() -> None:
    """零回归对照：门**关**时 ② 仍读 precision=（覆写在旧口径下依然有效）。

    这条是上面三条的正控——若门关时也与 precision 无关，说明新旧两条路径都没接对，
    上面的「无关性」就不构成任何证据。
    """
    ranked_low = _rank_episodes([_fact(_identity_episode(8.56), days_ago=1)], NOW, **RANK_KW)
    ranked_high = _rank_episodes([_fact(_identity_episode(72.0), days_ago=1)], NOW, **RANK_KW)
    from src.orchestration.memory_recall import normalized_importance

    lo = normalized_importance(ranked_low[0].content, 30.0)
    hi = normalized_importance(ranked_high[0].content, 30.0)
    assert lo < hi, "门关时 importance 必须随 precision 变化，否则本组测试失去参照"
