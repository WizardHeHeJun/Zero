"""方向判据升级 GATE（议会 2026-07-16 F2 修正3 升级版）。

议会二轮把方向判据从「n=10 手工探针点估计≥80%」升级为：
  留出集≥100 已知极性句、**Wilson 95% CI 下界≥0.70**（数学席：n=10 时 80% 的
  Wilson CI 下界仅 0.467，统计力度不足、不得终裁）。

已知极性锚点的构造（moderator 裁定，无需额外 rage/fear 标注）：EmoBank test split 中
writer-D 极端句即已知极性——writer-D 高（>阈值）预期 D_pred>0（高控制/趋近），
writer-D 低（<阈值）预期 D_pred<0（低控制/回避）。方向正确率=D_pred 符号与预期一致的比例。

诚实说明（不粉饰）：本判据测的是 D 头对 **writer-D 极端句**的留出集方向泛化（in-distribution），
比 gate_text_coping.py 的手工 rage/fear 语义探针更「容易」（后者更独立地测语义方向对齐）。
两者都报，以议会定的 Wilson CI 判据终裁，手工探针作补充参考。

用法：python -m scripts.direction_gate_text_coping --csv data/emobank_writer_filtered.csv \
  --weights artifacts/text_coping_regressor.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import math

import torch

from src.agents.models.text_affect_regressor_st import (
    DEFAULT_ENCODER,
    encode_texts,
    load_st_text_affect_regressor,
)

logger = logging.getLogger(__name__)

# 手工语义探针（复用 gate_text_coping 的补充参考；D>0=趋近/高控制，D<0=回避/低控制）
PROBES_HIGH: tuple[str, ...] = (
    "I will make him pay for this.",
    "I am in complete control of the situation.",
    "I can handle whatever comes my way.",
    "I refuse to back down.",
    "I will take care of it myself.",
)
PROBES_LOW: tuple[str, ...] = (
    "I am completely helpless.",
    "There is nothing I can do.",
    "I feel powerless to stop it.",
    "I am at their mercy.",
    "It is entirely out of my hands.",
)


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval 下界（n 次里 k 次正确）。"""
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / denom


def _read_test_extremes(
    csv_path: str, *, hi_thr: float, lo_thr: float
) -> tuple[list[str], list[str]]:
    """从 test split 取 writer-D 极端句（原始 1–5 量表）。返回 (高D句, 低D句)。"""
    hi: list[str] = []
    lo: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "test":
                continue
            d = float(row["D"])
            if d >= hi_thr:
                hi.append(row["text"])
            elif d <= lo_thr:
                lo.append(row["text"])
    return hi, lo


def _select_thresholds(csv_path: str, *, min_each: int) -> tuple[float, float, int, int]:
    """从最极端阈值向内走，选出令高/低两侧各 ≥min_each 的最极端 (hi_thr, lo_thr)。"""
    best: tuple[float, float, int, int] | None = None
    # 高侧从 4.5 向 3.5 放宽；低侧从 1.5 向 2.5 放宽（1–5 量表，3=中性）
    for step in range(0, 11):
        hi_thr = 4.5 - 0.1 * step
        lo_thr = 1.5 + 0.1 * step
        hi, lo = _read_test_extremes(csv_path, hi_thr=hi_thr, lo_thr=lo_thr)
        if len(hi) >= min_each and len(lo) >= min_each:
            return hi_thr, lo_thr, len(hi), len(lo)
        best = (hi_thr, lo_thr, len(hi), len(lo))
    assert best is not None
    return best  # 放到最宽仍不足 min_each，返回最宽阈值（调用方据实测数量决断）


def _direction_rate(
    texts_hi: list[str], texts_lo: list[str], weights: str, encoder: str
) -> tuple[int, int]:
    """D 头在极端句上的方向正确数与总数。高句期望 D_pred>0，低句期望 D_pred<0。"""
    model = load_st_text_affect_regressor(weights, output_dim=1)
    correct = 0
    with torch.no_grad():
        if texts_hi:
            hi_pred = model(encode_texts(texts_hi, encoder=encoder))[:, 0].tolist()
            correct += sum(1 for v in hi_pred if v > 0)
        if texts_lo:
            lo_pred = model(encode_texts(texts_lo, encoder=encoder))[:, 0].tolist()
            correct += sum(1 for v in lo_pred if v < 0)
    return correct, len(texts_hi) + len(texts_lo)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="方向判据升级 GATE（Wilson CI 下界≥0.70）")
    parser.add_argument("--csv", default="data/emobank_writer_filtered.csv")
    parser.add_argument("--weights", default="artifacts/text_coping_regressor.pt")
    parser.add_argument("--min-each", type=int, default=50, help="高/低两侧各最少句数")
    args = parser.parse_args()
    encoder = DEFAULT_ENCODER

    hi_thr, lo_thr, n_hi, n_lo = _select_thresholds(args.csv, min_each=args.min_each)
    texts_hi, texts_lo = _read_test_extremes(args.csv, hi_thr=hi_thr, lo_thr=lo_thr)
    logger.info(
        "阈值：writer-D≥%.1f(高,%d句) / ≤%.1f(低,%d句)；总 %d",
        hi_thr,
        n_hi,
        lo_thr,
        n_lo,
        n_hi + n_lo,
    )

    correct, total = _direction_rate(texts_hi, texts_lo, args.weights, encoder)
    lb = wilson_lower_bound(correct, total)
    rate = correct / total if total else float("nan")

    # 补充参考：手工语义探针
    p_correct, p_total = _direction_rate(list(PROBES_HIGH), list(PROBES_LOW), args.weights, encoder)

    logger.info("─" * 60)
    logger.info("方向判据（EmoBank 极端句留出集，议会终裁判据）：")
    logger.info(
        "  正确 %d/%d = %.1f%%；Wilson 95%% CI 下界 = %.4f（需≥0.70）",
        correct,
        total,
        rate * 100,
        lb,
    )
    logger.info(
        "  补充参考（手工语义探针 10 句）：%d/%d = %.0f%%",
        p_correct,
        p_total,
        p_correct / p_total * 100,
    )
    logger.info("─" * 60)
    if total < 2 * args.min_each:
        verdict = f"数据不足(每侧<{args.min_each})——阈值已放到最宽仍不够，须扩数据或降 min_each"
    elif lb >= 0.70:
        verdict = (
            f"PASS（Wilson 下界 {lb:.3f}≥0.70）→ 可解锁 PerceptionAgent 生产接线 + 精度可评上调"
        )
    else:
        verdict = f"FAIL（Wilson 下界 {lb:.3f}<0.70）→ 方向监督不足，暂不接生产/维持透传"
    logger.info("裁决：%s", verdict)
    print(f"DIRECTION-GATE: {correct}/{total} lb={lb:.4f} => {verdict}")


if __name__ == "__main__":
    main()
