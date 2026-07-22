"""记忆层通用纯函数（无 LLM、无 torch、确定性可复现）。

目前提供 Hill 饱和归一：normalize_precision(p, scale)。
被 consolidation.py（salience 代理）和 memory_recall.py（normalized_importance）共用，
消除 DRY——两处原先各自内联 p/(p+C) 逻辑。

Hill 归一与 Kalman 增益/逆方差加权同构（单调有界、边际递减），
C=scale 为半饱和常数；scale=30 与实测 precision 量级（~28–72）对齐（议会 D8/WARN-3b）。
"""

from __future__ import annotations


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
