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
_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "first_contact": re.compile(r" \| first_contact=True"),
    "commitment": re.compile(r" \| commitment=True"),
    "identity": re.compile(r" \| identity=\w+"),
}


def parse_importance_tags(content: str) -> dict[str, bool]:
    """解析 episode content 尾部元数据段里的语义重要性 tag，返回各 tag 是否命中。

    **位置锚定**，不是「取最后一个匹配」：先定位最后一个 ` | value=<数>`，只在其**之后**
    的子串里找 tag。返回 dict 恒含三个键（first_contact / commitment / identity）。

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
