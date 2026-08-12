"""记忆层通用纯函数（无 LLM、无 torch、确定性可复现）。

提供两个共用纯函数：
  - parse_importance(content)      — 从 episode 文本解析写入显著度 `precision=`。
  - normalize_precision(p, scale)  — Hill 饱和归一。

被 consolidation.py（salience 代理）和 memory_recall.py（normalized_importance）共用，
消除 DRY——两处原先各自内联 p/(p+C) 与 precision 正则逻辑。

Hill 归一与 Kalman 增益/逆方差加权同构（单调有界、边际递减），
C=scale 为半饱和常数；scale=30 与实测 precision 量级（~28–72）对齐（议会 D8/WARN-3b）。
"""

from __future__ import annotations

import re

_PRECISION_RE = re.compile(r"precision=([0-9]+(?:\.[0-9]+)?)")

# 可选 importance tag 的位置锚点：`precision=` 是**全部写入路径共同必拼、且位于所有可选
# tag 之前**的字段。只在**最后一个** `precision=` 之后的子串里找 tag——用户原话在 gist 段
# （位于系统元数据之前）够不到，即便原话里也写了 precision= 也只会把锚点前移、不会后移。
# ⚠ 锚点不能取 `value=`：`ChatDriver._maybe_seed_memories` 的种子记忆格式是
# `<text> | precision=.. | seed=True | first_contact=True`，**不含 value=** ⇒ 取 value=
# 会让种子记忆的 tag 全部**静默失效**（不驱红任何断言）。改锚点前须复核全部写入路径。
_TAG_ANCHOR_RE = re.compile(r"precision=-?[0-9]+(?:\.[0-9]+)?")
# 各 tag 的默认权重（**等权**）。0.2 使单 tag 命中时 I/b0 = 1.2，精确复现既有
# first_contact ×1.2。等权是刻意选择：三判据无共同度量单位，不得由文献效应量反推。
_DEFAULT_TAG_WEIGHT = 0.2

_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "first_contact": re.compile(r" \| first_contact=True"),
    "commitment": re.compile(r" \| commitment=True"),
    "identity": re.compile(r" \| identity=\w+"),
}


def importance_signal(
    tags: dict[str, bool],
    *,
    b0: float = 0.5,
    weights: dict[str, float] | None = None,
) -> float:
    """由确定性语义 tag 算 episode 重要性，noisy-OR 组合，值域 `[b0, 1)`。

        I = 1 − (1 − b0) · Π_k (1 − w_k)^{tag_k}

    **不接受 `affect_precision` 或任何后验精度作为入参**（PRD 目标 G2 的静态可检查形式）：
    后验精度衡量的是「引擎对情绪判断有多确定」，与「这条内容有多重要」方向常相反——
    闲聊「刚整理了一下桌子」归一后 0.374 压过心事「我最近状态不太好」0.268，
    身份自陈「我叫林川」仅 8.56。本函数只吃内容判据，不吃情绪确定性。

    为何用 noisy-OR 而非「加和 + clamp」（数学席 Q2）：多 tag 共现时加和会撞到硬上界 1.0，
    同一顶格值抹平「多重要」的区分度（identity∧commitment 与三 tag 全中会并列）；
    noisy-OR 渐近趋近 1 但永不相等，保序性更强。且它**天生有界**，不需要 Hill 归一那种
    为匹配经验量级而校准出来的 scale 常数（现状 `normalize_precision` 的 30 即由 precision
    实测 ~28–72 凑出）——避免重蹈 Stevens 1946 尺度类型误用。

    为何各 tag **等权**（心理席 Q1-b）：三判据分别以自由回忆准确率（自我参照效应）、
    再认反应时（意图优势效应）、印象评定量表（人际印象首因）测得，**无共同度量单位**，
    比较效应量来定序在方法论上不成立。`w` 默认 0.2 使单 tag 命中时 `I/b0 = 1.2`，
    **精确复现既有 `first_contact ×1.2`**——仓内唯一在跑且未被判定为误用的系数，
    非拍脑袋取值。调 `w` 会破坏这个锚定关系，已由单测钉死。

    ⚠ 这三个 tag 是神经机制的**工程代理**，不是机制本身：真实计算内容是新颖性预测误差 /
    图式一致性 / 奖赏情境（Tse 2007 图式一致性快速巩固与身份自陈对应最紧），
    而非关键词匹配。沿用 `consolidation.py` 模块头「salience 是 BLA-NE 唤醒调制的工程
    代理，非直接测量」的诚实标注风格。依据与引文见 `PRP/importance-signal/design.md`。

    参数：
      tags    — `parse_importance_tags` 的输出（缺键按未命中处理）。
      b0      — 无任何 tag 时的中性基线，默认 0.5，与 `parse_importance` 的「缺失 → 0.5」同源。
      weights — 各 tag 权重；缺省全部取 `_DEFAULT_TAG_WEIGHT`。
    """
    w_map = weights if weights is not None else dict.fromkeys(_TAG_PATTERNS, _DEFAULT_TAG_WEIGHT)
    residual = 1.0 - b0
    for name, hit in tags.items():
        if not hit:
            continue
        w = w_map.get(name, _DEFAULT_TAG_WEIGHT)
        residual *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - residual


def importance_excess(importance: float, b0: float = 0.5) -> float:
    """重要性相对中性基线的**超出比例** `u = (I − b0)/(1 − b0)`，钳到 [0, 1]。

    供 `_rank_episodes` 的衰减调制与 `EbbinghausDecay.a_eff` **共用**（两处若各写一份，
    会重演 parse_importance 当初两处副本、最后不得不上提的老路）。

    用途：把「以 0 为原点的幂函数」`I^κ` 换成「以中性基线为不动点」的线性混合
    `1 + κ·u`。`u=0`（无 tag / 中性）时该乘子对**任意 κ** 恒为 1 ⇒ 开关不触碰基线校准，
    正交性写进参数化本身而非靠注释约束（`ai-docs/pitfalls.md`「旋钮的副作用要写进
    参数化本身」，先例 MICRO_TREMOR_RATIO 叠加式→混合式）。

    ⚠ 为什么必须换掉 `I^κ`（2026-08-12 数学席判**失真**，主程复算证实且更严重）：
    `b0^κ ≠ 1`（`b0<1, κ>0` 时恒成立）⇒ **即便没有任何 tag 的普通 episode**，只要打开
    该门，其 recency 就被系统性压低。实测 Δt=4 天、κ=2 时闲聊典型 episode 压低 **94%**，
    三维占比由 `recency 31.8%/sim 52.4%/imp 15.9%` 变成 `2.8%/74.6%/22.6%`——加权和被
    悄悄变成「几乎只看 sim」。这与 I 有无噪声无关，是纯结构性的基线漂移。
    2026-08-11 那次离线 A/B 所报「高显著者上移 ⇒ 机制生效」为**误读**：真实成因是
    recency 维被整体压扁，当时 sim 被固定成同值，剩下起作用的只有 importance。
    """
    if b0 >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (importance - b0) / (1.0 - b0)))


def parse_importance_tags(content: str) -> dict[str, bool]:
    """解析 episode content 尾部元数据段里的语义重要性 tag，返回各 tag 是否命中。

    **位置锚定**，不是「取最后一个匹配」：先定位最后一个 `precision=<数>`，只在其**之后**
    的子串里找 tag。返回 dict 恒含三个键（first_contact / commitment / identity）。
    锚点取 `precision=` 而非 `value=` 的原因见 `_TAG_ANCHOR_RE` 上方注释（种子记忆格式
    不含 `value=`，取它会让那批 tag 静默失效）。

    ⚠ 为什么不能沿用 `parse_importance` 的「取最后一个匹配」口径（PRP 执行期发现·
    议会原表述不足）：`parse_importance` 安全是因为 `precision=` **系统必拼、总是存在**，
    最后一个匹配必属系统；而这三个 tag 都是**可选**的——系统本轮未打时，用户原话里的
    字面串就是**唯一**匹配，取最后一个照样命中 ⇒ 用户自称即可提权。防线必须建在
    **位置**上，不能建在匹配序号上。样本见 `tests/fixtures_importance_tags.py`。

    无锚点（空串 / 历史异常数据 / 无元数据段）→ 全 False，不猜测。
    """
    anchors = list(_TAG_ANCHOR_RE.finditer(content))
    if not anchors:
        return dict.fromkeys(_TAG_PATTERNS, False)
    tail = content[anchors[-1].end() :]
    return {name: pattern.search(tail) is not None for name, pattern in _TAG_PATTERNS.items()}


def parse_importance(content: str) -> float:
    """从 episode 文本解析写入时显著度 `precision=`；缺失/畸形返回 0.5 保守默认。

    对应 SupervisorAgent 固化的写入格式 `gist | ... | precision=<float> | ...`。取**最后一个**
    precision= 匹配——结构化元数据字段在尾部、用户原话/语言在前段，避免用户输入里的
    `precision=0.99` 污染评分（WARN-1）。纯正则、无 LLM（守确定性热路径 BLOCK-1）。
    返回 0 会让 γ 维恒失效，故缺失取 0.5 中性值。

    此前 memory_recall.parse_importance 与 consolidation 内联的 _parse_importance_local
    各持一份同逻辑副本；上提至记忆层 utils 统一（orchestration → memory 为合法下调）。
    """
    matches = _PRECISION_RE.findall(content)
    if not matches:
        return 0.5
    try:
        return float(matches[-1])
    except ValueError:
        return 0.5


def normalize_precision(p: float, scale: float = 30.0) -> float:
    """Hill 饱和归一：p / (p + scale)，结果 ∈ [0, 1)。

    用于把无界的 affect_precision（方差倒数，实测 ~28–72）
    归一到与 sim/recency 同量纲的 (0, 1) 区间。

    参数：
      p     — 原始 precision 值（≥ 0）。
      scale — 半饱和常数 C（默认 30.0，与 memory_recall.py 实测量级对齐；
              scale ≤ 0 时回退为 0.0 防除零）。

    返回值 ∈ [0, 1)：
      p=0.5 (fallback) → ~0.016
      p=28             → ~0.483
      p=30             → 0.500（半饱和点）
      p=72             → ~0.706（× 0.5 后 ≈ 0.353，过 salience 门 0.25）

    限制声明（神经席 WARN-3b）：
      normalize_precision 不区分高/低 RPE episode 的意外度差异——
      所有 episode 获同等意外度权重（rpe=0.5 常数代理，丢失了 BLA-NE
      唤醒调制中因突发/意外事件引起的差异化巩固效应）。
    """
    if scale <= 0.0:
        return 0.0
    return p / (p + scale)
