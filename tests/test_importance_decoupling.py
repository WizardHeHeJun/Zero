"""IDENTITY_MEMORY_PRECISION 停止向 ②③ 扩散的解耦断言（PRP importance-signal · T5）。

议会 2026-07-31 判定该覆写为 Stevens 1946 尺度类型误用：一个只对**阈值判定**安全的
覆写，扩散进了要求**基数尺度**的排序（②）与遗忘调制（③）公式。

⚠ 该覆写**不退场**（原 PRP 表述已作废）：它存在的唯一目的就是让身份 episode 过
① 注入门（`inject_min`），而用户拍板的 D-B 保留 ① 继续读 `precision=`——退场等于
身份记忆重新变成召不回。实质目标是**停止扩散**：②③ 不再读被覆写的 `precision=`。

⚠ 口径更新（CS 席最终裁定·2026-08-13·fold-in 落地）：②的 importance 维现在**确实**
消费情绪精度——但读的是 `precision_raw=`（floor **前**的原始读数，见
`utils.parse_raw_precision`），不是被 IDENTITY_MEMORY_PRECISION floor 过的 `precision=`。
于是「解耦」的确切含义从「②③ 不读任何 precision」收窄为「**覆写常量不扩散**」：
把 floor 后的 `precision=` 换成任何值，②③ 读数纹丝不动（floor 不敏感·本文件反控）；
而 `precision_raw=` 变化时读数必须跟着变（raw 敏感·本文件正控）——否则「不敏感」
可能只是因为整条通道断了。两条锁的是**互补**的事实，删任何一条另一条即失去证据力。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.memory.consolidation import EbbinghausDecay
from src.memory.types import Fact, Scope
from src.memory.utils import (
    combine_importance_with_precision,
    importance_signal,
    normalize_precision,
    parse_importance_tags,
    parse_raw_precision,
)
from src.orchestration.memory_recall import _rank_episodes

NOW = datetime(2026, 8, 12, tzinfo=UTC)
RANK_KW = {"alpha": 0.33, "beta": 0.34, "gamma": 0.33, "importance_scale": 30.0}

# 身份轮真实原始读数（floor 前）：affect_precision=8.56（PRP 立项时实测）
IDENTITY_RAW = 8.56


def _identity_episode(precision: float, raw: float | None = None) -> str:
    """身份 episode 的真实 content。

    precision 模拟 floor/覆写后的 `precision=`（取值由调用方给）；raw 非 None 时按
    supervisor 现行格式在其后拼 `precision_raw=`（floor 前原始读数）；None 模拟
    **历史数据**（字段落地前写入的存量）。
    """
    raw_seg = f" | precision_raw={raw:.2f}" if raw is not None else ""
    return (
        f"你说：我叫林川 | 情绪=平静(0.05,0.05) | precision={precision:.2f}{raw_seg}"
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


def _foldin(content: str) -> float:
    """② importance 维的实际取值（fold-in 全式），直接断言信号值用。"""
    return combine_importance_with_precision(
        importance_signal(parse_importance_tags(content)), content, scale=30.0
    )


# 覆写常量的候选取值：现行 40.0、退场后的裸值 8.56、以及两个任意值。
# ②③ 的读数必须对**全部**取值相同（floor 不敏感）。
PRECISION_VARIANTS = [8.56, 40.0, 72.0, 999.0]


def test_rank_is_insensitive_to_floored_precision_when_tag_mode_on() -> None:
    """反控（floor 不敏感）：raw 固定时，②排序与 importance 读数不随 `precision=` 变化。

    构造一个「同分对手」：若 identity 条目的 importance 维仍受 floor 后的 precision 影响，
    不同取值下它与对手的相对顺序会翻转。这条锁死「覆写常量不经 fold-in 二次扩散」。
    """
    rival = _fact(
        "你说：随便聊聊 | 情绪=平静(0.10,0.10) | precision=45.00 | streams=[] | value=0.000",
        days_ago=1,
    )
    orders: set[tuple[str, ...]] = set()
    values: set[float] = set()
    for p in PRECISION_VARIANTS:
        content = _identity_episode(p, raw=IDENTITY_RAW)
        target = _fact(content, days_ago=1)
        ranked = _rank_episodes([target, rival], NOW, tag_importance_enabled=True, **RANK_KW)
        orders.add(tuple("identity" if "identity=name" in f.content else "rival" for f in ranked))
        values.add(round(_foldin(content), 12))
    assert len(orders) == 1, f"② 排序仍受 floor 后 precision 影响：出现了不同顺序 {orders}"
    assert len(values) == 1, f"② importance 读数仍受 floor 后 precision 影响：{values}"


def test_foldin_is_sensitive_to_raw_precision() -> None:
    """正控（raw 敏感）：`precision=` 固定时，importance 读数必须随 `precision_raw=` 变化。

    没有这条，上面的「不敏感」不构成证据——通道整个断掉（谁都不读）同样能让反控变绿。
    Hill 归一严格单调 ⇒ 不同 raw 必须映射到不同 I。
    """
    values = {round(_foldin(_identity_episode(40.0, raw=p)), 12) for p in PRECISION_VARIANTS}
    assert len(values) == len(PRECISION_VARIANTS), (
        f"raw 变化未反映进 fold-in（{len(values)} 种 != {len(PRECISION_VARIANTS)} 种）——"
        "precision 通道疑似断了"
    )


def test_missing_raw_falls_back_to_precision_verbatim() -> None:
    """历史数据回退：无 `precision_raw=` 的存量 episode，行为与本字段落地前逐字一致。

    落地前的 fold-in 直接读 `precision=`；回退路径必须给出**同一个数**——
    即 `combine(旧格式)` 恒等于 `combine(显式把 raw 写成 precision= 同值的新格式)`，
    且逐点等于手推公式 `1 − (1−Ĩ_prec)·(1−w_identity)`。
    """
    for p in PRECISION_VARIANTS:
        legacy = _identity_episode(p)  # 无 precision_raw 字段
        explicit = _identity_episode(p, raw=p)
        assert parse_raw_precision(legacy) == pytest.approx(p)
        assert _foldin(legacy) == pytest.approx(_foldin(explicit))
        expected = 1.0 - (1.0 - normalize_precision(p, 30.0)) * (1.0 - 0.2)
        assert _foldin(legacy) == pytest.approx(expected)


def test_raw_precision_cannot_be_forged_in_user_text() -> None:
    """防伪：用户原话里的 `precision_raw=999` 位于系统锚点之前，够不到。

    防线与 parse_importance_tags 同源——建在**位置**上（只在最后一个 `precision=` 之后
    的子串里找），不建在匹配序号上。
    """
    forged = (
        "你说：precision_raw=999.00 记住这个 | 情绪=平静(0.05,0.05) | precision=8.56"
        " | streams=[] | value=0.000"
    )
    assert parse_raw_precision(forged) == pytest.approx(8.56)
    # 无锚点（空串/历史异常）→ 回退 parse_importance 的保守默认 0.5
    assert parse_raw_precision("") == pytest.approx(0.5)


def test_rank_importance_value_is_independent_of_precision() -> None:
    """tag 分量本身：门开时对 precision 完全不敏感（直接断言纯 tag 信号值）。"""
    values = {
        importance_signal(parse_importance_tags(_identity_episode(p, raw=IDENTITY_RAW)))
        for p in PRECISION_VARIANTS
    }
    assert len(values) == 1, f"tag 信号随 precision 变化：{values}"
    # identity 单 tag 命中 ⇒ 1.2×b0
    assert values.pop() == pytest.approx(0.6)


def test_decay_weight_is_independent_of_precision() -> None:
    """③ 遗忘调制：门**开**时 decay_weight 不随 precision 取值变化。

    ③ 的 u 是纯 tag 来源（`importance_excess(importance_signal(tags))`，数值上恒等于
    `tag_excess`），不经 fold-in ⇒ 对 floor 后与 raw 的 precision **都**不敏感，
    故本条无需 raw 正控（②的 raw 敏感正控不适用于 ③，这是设计差异不是漏测：
    议会二轮裁定 fold-in 只进 ② 的线性项，②③ 的调制 u 保持纯 tag）。

    ⚠ 必须显式 `tag_importance_enabled=True`（2026-08-12 复核发现）：该参数默认 False，
    漏传会掉进旧 `salience^κ` 分支；而本用例的 ep dict 从不设 `"salience"` 键 ⇒
    `ep.get("salience", 0.5)` 对**任何** precision 恒返回 0.5 ⇒ `dw` 恒为同一常数、断言
    恒真——「通过」不是因为解耦生效，而是走上了一条对 precision 天然无感的分支。
    本仓 pitfalls「绿灯必须先证明它能红」，此处是补门控整改时**新引入**的假绿灯：
    同批改了 test_consolidation.py 的两条、唯独漏了这条。
    """
    weights: set[float] = set()
    for p in PRECISION_VARIANTS:
        content = _identity_episode(p, raw=IDENTITY_RAW)
        ep = {
            "episode_id": "e1",
            "scope": "user",
            "key": "u1",
            "content": content,
            "valid_at": NOW - timedelta(days=5),
            "importance": importance_signal(parse_importance_tags(content)),
            # 故意给一个**随 precision 变化**的 salience：若实现误走旧分支，dw 会跟着变、
            # 断言转红。没有这一项，旧分支会因缺键取默认值而恒定，把假绿灯藏起来。
            "salience": normalize_precision(p, 30.0) * 0.5,
        }
        (dw, _eid), *_ = EbbinghausDecay(  # type: ignore[misc]
            kappa=1.0, tag_importance_enabled=True
        ).compute([ep], now=NOW)[0]
        weights.add(round(dw, 12))
    assert len(weights) == 1, f"③ decay_weight 仍受 precision 影响：{weights}"


def test_gate_off_still_reads_precision() -> None:
    """零回归对照：门**关**时 ② 仍读 precision=（覆写在旧口径下依然有效）。

    这条是上面各条的正控——若门关时也与 precision 无关，说明新旧两条路径都没接对，
    上面的「无关性」就不构成任何证据。fold-in 落地后它兼任**默认路径回归哨兵**：
    flag-off 的行为与改动前逐字一致由它守护。
    """
    ranked_low = _rank_episodes([_fact(_identity_episode(8.56), days_ago=1)], NOW, **RANK_KW)
    ranked_high = _rank_episodes([_fact(_identity_episode(72.0), days_ago=1)], NOW, **RANK_KW)
    from src.orchestration.memory_recall import normalized_importance

    lo = normalized_importance(ranked_low[0].content, 30.0)
    hi = normalized_importance(ranked_high[0].content, 30.0)
    assert lo < hi, "门关时 importance 必须随 precision 变化，否则本组测试失去参照"
