"""CS 席 Q6 硬性前置：真实失真样本对的回归锚点（PRP importance-signal · T7）。

样本不是臆造的对抗用例，而是 2026-08-11 离线复算里**真实记录**的失真现场：

| 输入 | affect_precision 归一后 | 实际重要性 |
| --- | --- | --- |
| 「刚整理了一下桌子」 | 0.374 | 低 |
| 「我最近状态不太好，你觉得是因为什么？」 | 0.268 | 高 |

情绪越平淡确定 → 后验精度越高；内容越重要但情绪越模糊 → 精度反而越低。这是
`affect_precision` 当重要性代理的**方向性错误**，也是本 PRP 的立项理由。

⚠ 锚点口径二次修订（议会二轮·2026-08-13·fold-in 落地）：上一版「双双落 b0 并列」
只对**纯 tag 分量**成立。438 轮实测证明「替代」把 87.3% 无 tag 的 episode 全压到 b0，
importance 维取值从 171 种塌成 2 种、排序被 sim 独占——改为 fold-in 后，无 tag episode
精确退化为 precision 基线 ⇒ **这对样本的方向错误在 ② 全量 importance 上原样回归**。
这不是回归缺陷，是 identifiability 边界（任何对 Ĩ_prec 单调的 (tag, precision) 方案都
翻转不了两个无 tag 样本的相对序，见 `combine_importance_with_precision` docstring）；
议会裁定锚点**改测 fold-in 能保证的两条**：「不塌缩」与「tag 优势」（见文末两条）。
原「方向不反」断言保留在**纯 tag 分量**层级，并显式钉死全量层级的取舍后果——
两层一起读才是完整图景，删掉任何一层都会把取舍粉饰成全胜。
"""

from __future__ import annotations

import pytest

from src.memory.utils import (
    combine_importance_with_precision,
    importance_signal,
    normalize_precision,
    parse_importance_tags,
)
from src.orchestration.memory_recall import normalized_importance

# 2026-08-11 离线复算的真实 content（precision 取当时实测值）
CHORE = "你说：刚整理了一下桌子 | 情绪=平静(0.10,0.10) | precision=28.00 | streams=[] | value=0.000"
DISTRESS = (
    "你说：我最近状态不太好，你觉得是因为什么？ | 情绪=不悦(-0.30,0.20) | precision=11.00"
    " | streams=[] | value=-0.150"
)


def _foldin(content: str) -> float:
    """② importance 维的实际取值（fold-in 全式）。"""
    return combine_importance_with_precision(
        importance_signal(parse_importance_tags(content)), content, scale=30.0
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


def test_tag_component_no_longer_ranks_them_backwards() -> None:
    """纯 tag 分量：闲聊**不再大于**心事（两条都无 tag ⇒ 双双落 b0 ⇒ 并列）。

    断言写作 `<=` 而非 `<`——这一层消除了错误的区分，但没有建立正确的区分。
    真正覆盖无 tag 内容依赖后续 PRP 的写入门 `is_informative` 通道。
    """
    chore = importance_signal(parse_importance_tags(CHORE))
    distress = importance_signal(parse_importance_tags(DISTRESS))
    assert chore <= distress, f"tag 分量仍把闲聊排在心事之前：{chore} > {distress}"
    assert chore == pytest.approx(0.5)
    assert distress == pytest.approx(0.5)


def test_foldin_knowingly_retains_the_backwards_pair() -> None:
    """⚠ 显式钉死取舍：fold-in 全量 importance 上，这对样本的方向错误**原样保留**。

    `28→0.483 > 11→0.268`——与旧口径同向。这是议会二轮**知情接受**的 identifiability
    边界（换取 87.3% 无 tag episode 不塌缩），不是待修 bug。若本断言转红（方向翻转了），
    说明有人往 fold-in 里塞了超出 (tag, precision) 输入空间的东西——那须重过设计门，
    而不是顺手庆祝。
    """
    assert _foldin(CHORE) > _foldin(DISTRESS)


def test_tagged_content_does_get_separated() -> None:
    """能力边界的正面证明：**带 tag** 的内容确实被抬高，信号本身有区分力。

    与上面「无 tag 并列」并读，才是完整图景——不是信号没用，是它只覆盖有 tag 的内容。
    （样本 2026-08-13 曾因 w_commitment 临时置 0 从 commitment 换成 identity；
    2026-08-14 权重已凭实跑占比表恢复，样本保留 identity 不回换——两 tag 等权，
    断言语义不变，沿革见 test_importance_tags 的
    test_signal_commitment_weight_restored_with_evidence。）
    """
    distress_with_identity = DISTRESS + " | identity=selfstate"
    plain = importance_signal(parse_importance_tags(DISTRESS))
    tagged = importance_signal(parse_importance_tags(distress_with_identity))
    assert tagged > plain
    assert tagged / plain == pytest.approx(1.2)


# ── 议会二轮裁定的两条替代锚点（张力 1：取代「只测方向不反」的单层锚点） ──────────


def test_foldin_does_not_collapse_precision_distribution() -> None:
    """锚点①「不塌缩」：171 种 precision 取值须映射到 ≥100 种不同 I（议会给的下限）。

    438 轮实测的回归形态正是塌缩：门开后 importance 取值 171 种 → 2 种、维影响力跨度
    0.213 → 0.033、排序被 sim 独占。Hill 归一严格单调 + w_p=1.0 精确退化 ⇒ 实际应
    **全部**不同（断言取严格式；若未来实现引入分箱/量化，至少不得跌破议会下限 100）。
    """
    values: set[float] = set()
    n = 171
    for i in range(n):
        p = 4.0 + i * 0.5  # 覆盖实测量级 ~4–89
        content = (
            f"你说：普通一句话 | 情绪=平静(0.10,0.10) | precision={p:.2f}"
            f" | precision_raw={p:.2f} | streams=[] | value=0.000"
        )
        values.add(round(_foldin(content), 12))
    assert len(values) == n, f"fold-in 塌缩：{n} 种输入只剩 {len(values)} 种 I"
    assert len(values) >= 100  # 议会明文下限，独立于上一行的严格式


def test_foldin_tag_advantage() -> None:
    """锚点②「tag 优势」：命中 tag 的低 precision 句能反超无 tag 的高 precision 句。

    结构保证（全域成立）：同 raw 下带 tag 严格大于不带（`I = 1−(1−Ĩ)·Π`，`Π<1`）。
    反超（跨 precision）是**有界**的：`I_tag(lo) > Ĩ(hi) ⇔ Ĩ(hi) < w + (1−w)·Ĩ(lo)`
    ——单 tag w=0.2、raw=8.56（Ĩ=0.222）时边界为 Ĩ(hi) < 0.378，即 p_hi ≲ 18.2。
    样本取真实身份轮 raw=8.56 对普通句 raw=12.0（边界内·反超）与 raw=65.66
    （findings 里的陈述句实测值·边界外·不反超）——**两侧都钉死**，防止有人把
    「有界反超」读成「tag 全域优先」再据此调权重。
    """
    identity_lo = (
        "你说：我叫林川 | 情绪=平静(0.05,0.05) | precision=40.00 | precision_raw=8.56"
        " | streams=[] | value=0.000 | identity=name"
    )
    plain_mid = (
        "你说：今天有点忙 | 情绪=平静(0.10,0.10) | precision=12.00 | precision_raw=12.00"
        " | streams=[] | value=0.000"
    )
    plain_hi = (
        "你说：现在号上四千多万了 | 情绪=平静(0.10,0.10) | precision=65.66"
        " | precision_raw=65.66 | streams=[] | value=0.000"
    )
    # 同 raw：tag 严格优势（结构性，任意 raw 成立）
    same_raw_plain = (
        "你说：普通一句话 | 情绪=平静(0.05,0.05) | precision=8.56 | precision_raw=8.56"
        " | streams=[] | value=0.000"
    )
    assert _foldin(identity_lo) > _foldin(same_raw_plain)
    # 跨 raw·边界内：低 precision 带 tag 反超中等 precision 无 tag
    assert _foldin(identity_lo) > _foldin(plain_mid), "tag 优势未生效（边界内应反超）"
    # 跨 raw·边界外：不反超——有界性本身也是锚点的一部分
    assert _foldin(identity_lo) < _foldin(plain_hi), (
        "tag 优势越界（w=0.2 不该盖过 Ĩ=0.686 的强 precision 证据）"
    )
    # 数值锚：I_tag(8.56) = 1 − (1−0.222)·0.8
    expected = 1.0 - (1.0 - normalize_precision(8.56, 30.0)) * 0.8
    assert _foldin(identity_lo) == pytest.approx(expected)
