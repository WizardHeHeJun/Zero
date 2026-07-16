"""文本→coping_potential 独立性 GATE 评估（议会 2026-07-15 修正3+4·相 1 闸门）。

议会裁定 D 头训后须过硬门控方可接线（否则落选项 C 归档）：
  修正4 独立性：留出集上 r(V_pred, D_pred) 分级——
    ≤0.45 通过 / 0.45–0.50 警告(精度减半) / >0.50 BLOCK 落 C。
    V_pred 取 v0.1 VA 头（artifacts/text_affect_regressor_st.pt, output_dim=2）的 V 分量；
    D_pred 取本脚本训的独立 D 头（output_dim=1）。
  修正3 方向：已知极性 rage/fear 探针句上 D_pred 方向正确率 ≥80%（趋近/高控制→D>0；
    回避/低控制→D<0）为所有区间前提必要条件。

用 EmoBank 自带 split 列做留出（train 训 / test 评）；D 头仅 MLP、encoder 冻结（CPU 可行）。
本脚本是**评估闸**：跑完打印 GATE 裁决（PASS→可进相 2 接线 / FAIL→落 C）。不写运行时接线。

用法：
  python -m scripts.gate_text_coping --csv data/emobank_writer_filtered.csv \
    --va-weights artifacts/text_affect_regressor_st.pt --out artifacts/text_coping_regressor.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import torch
from torch import nn

from src.agents.affect_math import clamp
from src.agents.models.text_affect_regressor_st import (
    DEFAULT_ENCODER,
    STTextAffectRegressor,
    encode_texts,
    load_st_text_affect_regressor,
)

logger = logging.getLogger(__name__)

# 修正3 方向探针：极性已知（D>0=高控制/趋近，D<0=低控制/回避）。英文对齐 EmoBank 语域。
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


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else float("nan")


def _read_split(csv_path: str) -> tuple[list[str], list[float], list[str], list[float]]:
    """按 split 列切 train/test，返回 (tr_texts, tr_D, te_texts, te_D)；D 已归一。"""
    tr_t: list[str] = []
    tr_d: list[float] = []
    te_t: list[str] = []
    te_d: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = clamp((float(row["D"]) - 3.0) / 2.0, -1.0, 1.0)
            if row["split"] == "test":
                te_t.append(row["text"])
                te_d.append(d)
            else:  # train + dev 都进训练
                tr_t.append(row["text"])
                tr_d.append(d)
    if not tr_t or not te_t:
        raise ValueError("train/test 至少一侧为空——检查 split 列")
    return tr_t, tr_d, te_t, te_d


def train_d_head(
    train_texts: list[str], train_d: list[float], *, epochs: int, lr: float, encoder: str
) -> STTextAffectRegressor:
    """在 train 集训 output_dim=1 的独立 D 头（encoder 冻结，仅 MLP）。"""
    x = encode_texts(train_texts, encoder=encoder)
    y = torch.tensor([[v] for v in train_d], dtype=torch.float32)
    model = STTextAffectRegressor(dim=x.shape[1], encoder=encoder, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            logger.info("epoch %d D-head loss %.6f (n=%d)", epoch, float(loss.item()), x.shape[0])
    model.eval()
    return model


def evaluate_gate(
    csv_path: str, va_weights: str, out_path: str, *, epochs: int, lr: float
) -> dict[str, float]:
    """训 D 头 + 留出集独立性 r(V_pred,D_pred) + 方向正确率。返回 GATE 指标。"""
    encoder = DEFAULT_ENCODER
    tr_t, tr_d, te_t, te_d = _read_split(csv_path)
    logger.info("split: train+dev=%d, test=%d", len(tr_t), len(te_t))

    d_head = train_d_head(tr_t, tr_d, epochs=epochs, lr=lr, encoder=encoder)
    va_head = load_st_text_affect_regressor(va_weights, output_dim=2)  # v0.1 VA 头

    # 留出集：编码一次，V_pred 取 VA 头 [:,0]，D_pred 取 D 头 [:,0]
    with torch.no_grad():
        feats = encode_texts(te_t, encoder=encoder)
        v_pred = va_head(feats)[:, 0].tolist()
        d_pred = d_head(feats)[:, 0].tolist()
    r_vd = _pearson(v_pred, d_pred)
    r_label = _pearson([clamp(v, -1, 1) for v in v_pred], te_d)  # 参考：D_pred 追不追标注

    # 修正3 方向：探针
    with torch.no_grad():
        hi = d_head(encode_texts(list(PROBES_HIGH), encoder=encoder))[:, 0].tolist()
        lo = d_head(encode_texts(list(PROBES_LOW), encoder=encoder))[:, 0].tolist()
    correct = sum(1 for v in hi if v > 0) + sum(1 for v in lo if v < 0)
    direction_rate = correct / (len(hi) + len(lo))

    # 保存 D 头（权重无害·gitignore；接线与否由裁决定）
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d_head.state_dict(), out)

    return {
        "r_v_dpred": r_vd,
        "r_v_dlabel": r_label,
        "direction_rate": direction_rate,
        "n_test": float(len(te_t)),
    }


def _verdict(r_vd: float, direction_rate: float) -> str:
    """按 design.md 分级表 + 方向必要条件给裁决字符串。"""
    if direction_rate < 0.8:
        return f"FAIL(方向 {direction_rate:.0%}<80% → 落选项 C)"
    if r_vd <= 0.38:
        return "PASS(r≤0.38 强通过，可进相 2)"
    if r_vd <= 0.45:
        return "PASS(0.38<r≤0.45 噪声容差内，可进相 2)"
    if r_vd <= 0.50:
        return "WARN(0.45<r≤0.50 轻度共线放大 → 精度减半+披露后可进相 2)"
    return "FAIL(r>0.50 失独立性 → 落选项 C，不接内核)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="文本→coping_potential 独立性 GATE（修正3+4）")
    parser.add_argument("--csv", default="data/emobank_writer_filtered.csv")
    parser.add_argument("--va-weights", default="artifacts/text_affect_regressor_st.pt")
    parser.add_argument("--out", default="artifacts/text_coping_regressor.pt")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    m = evaluate_gate(args.csv, args.va_weights, args.out, epochs=args.epochs, lr=args.lr)
    verdict = _verdict(m["r_v_dpred"], m["direction_rate"])
    logger.info("─" * 60)
    logger.info("GATE 留出集(test, n=%d)：", int(m["n_test"]))
    logger.info("  独立性 r(V_pred, D_pred) = %.4f（阈值分级 0.45/0.50）", m["r_v_dpred"])
    logger.info("  参考 r(V_pred, D_label)  = %.4f", m["r_v_dlabel"])
    logger.info("  方向正确率(探针 10 句)   = %.0f%%（需≥80%%）", m["direction_rate"] * 100)
    logger.info("─" * 60)
    logger.info("GATE 裁决：%s", verdict)
    print(f"GATE: r(V,Dpred)={m['r_v_dpred']:.4f} dir={m['direction_rate']:.0%} => {verdict}")


if __name__ == "__main__":
    main()
