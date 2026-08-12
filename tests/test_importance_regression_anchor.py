"""CS 席 Q6 硬性前置：真实失真样本对的回归锚点（PRP importance-signal · T7）。

样本不是臆造的对抗用例，而是 2026-08-11 离线复算里**真实记录**的失真现场：

| 输入 | affect_precision 归一后 | 实际重要性 |
| --- | --- | --- |
| 「刚整理了一下桌子」 | 0.374 | 低 |
| 「我最近状态不太好，你觉得是因为什么？」 | 0.268 | 高 |

情绪越平淡确定 → 后验精度越高；内容越重要但情绪越模糊 → 精度反而越低。这是
`affect_precision` 当重要性代理的**方向性错误**，也是本 PRP 的立项理由。

⚠ **断言口径**（设计门裁定的固有边界，勿"加强"）：新口径下这两条**都不命中任何 tag**，
双双落中性 `b0` ⇒ 结果是**并列**（0.5 == 0.5），不是「后者严格大于前者」。
即：`0.374 > 0.268`（错误方向）→ `0.5 == 0.5`（无区分）。
**错误的区分被消除，正确的区分并未建立**——把断言写成「后者严格大于前者」是伪造绿灯，
做不到。真正覆盖无 tag 内容依赖后续 PRP 的「第一人称状态陈述」判据。
"""

from __future__ import annotations

import pytest

from src.memory.utils import importance_signal, normalize_precision, parse_importance_tags
from src.orchestration.memory_recall import normalized_importance

# 2026-08-11 离线复算的真实 content（precision 取当时实测值）
CHORE = "你说：刚整理了一下桌子 | 情绪=平静(0.10,0.10) | precision=28.00 | streams=[] | value=0.000"
DISTRESS = (
    "你说：我最近状态不太好，你觉得是因为什么？ | 情绪=不悦(-0.30,0.20) | precision=11.00"
    " | streams=[] | value=-0.150"
)


def test_old_proxy_ranks_them_backwards() -> None:
    """**正控**：旧口径（affect_precision 代理）在这对样本上给出**相反**的方向。

    这条必须绿——它证明本文件的锚点确实指向一个真实缺陷。若它转绿失败（即旧口径
    本来就没问题），下面那条「新口径不再反向」就不构成任何证据（恒真绿灯）。
    """
    chore = normalized_importance(CHORE, 30.0)
    distress = normalized_importance(DISTRESS, 30.0)
    assert chore == pytest.approx(28 / 58, abs=1e-3)
    assert distress == pytest.approx(11 / 41, abs=1e-3)
    assert chore > distress, "旧口径本应把闲聊排在心事之前（这正是要修的缺陷）"


def test_new_signal_no_longer_ranks_them_backwards() -> None:
    """新口径：闲聊**不再大于**心事。

    两条都无 tag ⇒ 双双落 b0 ⇒ 并列。断言写作 `<=` 而非 `<`——见模块 docstring 的
    口径说明：这一版消除了错误的区分，但没有建立正确的区分。
    """
    chore = importance_signal(parse_importance_tags(CHORE))
    distress = importance_signal(parse_importance_tags(DISTRESS))
    assert chore <= distress, f"新口径仍把闲聊排在心事之前：{chore} > {distress}"
    assert chore == pytest.approx(0.5)
    assert distress == pytest.approx(0.5)


def test_new_signal_is_insensitive_to_the_misleading_precision() -> None:
    """新信号对这两条的 `precision=` 差异（28 vs 11）完全不敏感——错配源被切断。"""
    assert importance_signal(parse_importance_tags(CHORE)) == importance_signal(
        parse_importance_tags(DISTRESS)
    )
    # 而旧口径对同一差异高度敏感（正控）
    assert normalize_precision(28.0, 30.0) != normalize_precision(11.0, 30.0)


def test_tagged_content_does_get_separated() -> None:
    """能力边界的正面证明：**带 tag** 的内容确实被抬高，说明信号本身有区分力。

    与上面「无 tag 并列」并读，才是完整图景——不是信号没用，是它只覆盖有 tag 的内容。
    """
    distress_with_commitment = DISTRESS + " | commitment=True"
    plain = importance_signal(parse_importance_tags(DISTRESS))
    tagged = importance_signal(parse_importance_tags(distress_with_commitment))
    assert tagged > plain
    assert tagged / plain == pytest.approx(1.2)
