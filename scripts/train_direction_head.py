"""符号监督训练 motivational_direction_prior 头 + 分侧方向门（议会三轮 2026-07-16）。

议会裁：换 anger(+)/fear(−) **离散符号监督**替代 SAM Dominance 幅度回归——交叉熵直罚
符号翻转（MSE 给不了）。本脚本落地议会 4 必改中的训练侧：
  - **Tanh 修复（CS 硬要求2）**：头**不带末层 Tanh**，用 BCEWithLogitsLoss on raw logit
    （末层 Tanh + BCEWithLogitsLoss 会注错误梯度）；推理时 tanh(logit) 映回 [−1,1]。
  - **软标签（Q4 神经席首选）**：anger→0.9 / fear→0.1（留 fight/freeze 噪声空间），
    另跑硬标签(1.0/0.0)基线对比。
  - **类不平衡（数学席）**：pos_weight=n_fear/n_anger 平衡多数 anger。
  - **分侧方向门（Q6/数学席）**：anger 侧、fear 侧**分别**报方向正确率 + Wilson CI 下界，
    每侧 n≥50、下界≥0.70 方过（不合并、防高分侧掩盖低分侧）。

本轮：GoEmotions 单飞（备选 A·Apache-2.0 干净·license 无传染）。跨源 OOD（crowd-enVENT +
EmpatheticDialogues）留待多源脚本。命名用 motivational_direction_prior（议会正名）。

用法：python -m scripts.train_direction_head --data-dir data/raw/goemotions \
  --out artifacts/motivational_direction_prior.pt [--hard]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import torch
from torch import nn

from scripts._train_common import write_provenance
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, encode_texts

logger = logging.getLogger(__name__)

ANGER_ID = "2"  # emotions.txt 顺序：anger=2
FEAR_ID = "14"  # fear=14


def _read_goemotions_signed(path: str) -> tuple[list[str], list[int]]:
    """读 GoEmotions TSV，取含 anger 且不含 fear→+1、含 fear 且不含 anger→0（BCE 类）。"""
    texts: list[str] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 2:
                continue
            ids = set(row[1].split(","))
            has_a, has_f = ANGER_ID in ids, FEAR_ID in ids
            if has_a and not has_f:
                texts.append(row[0])
                labels.append(1)  # anger=正类(+)
            elif has_f and not has_a:
                texts.append(row[0])
                labels.append(0)  # fear=负类(−)
    return texts, labels


def _read_crowdenvent_signed(path: str) -> tuple[list[str], list[int]]:
    """读 crowd-enVent_generation.tsv：emotion==anger→1(+)、fear→0(−)，text=generated_text。

    crowd-enVENT 是第一人称事件叙事域（补 GoEmotions 缺的叙事域 anger，治 anger 跨源掉分）。
    """
    texts: list[str] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            emo = row["emotion"].strip().lower()
            if emo == "anger":
                texts.append(row["generated_text"])
                labels.append(1)
            elif emo == "fear":
                texts.append(row["generated_text"])
                labels.append(0)
    return texts, labels


def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / denom


class DirectionHead(nn.Module):
    """句向量 → 单 logit（无 Tanh；BCEWithLogitsLoss 训练；推理 tanh(logit)∈[−1,1]）。

    ⚠训练侧保留此定义；**推理封装权威版在 src/agents/models/direction_head.py**（W1 生产接线用）。
    改 MLP 结构（hidden/层数）须两处同步，否则权重键形不兼容（code-review WARN-5）。
    """

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logit


def train_and_gate(
    data_dir: str,
    out_path: str,
    *,
    hard: bool,
    epochs: int,
    lr: float,
    crowdenvent_path: str | None = None,
    seed: int = 0,
) -> dict:
    """训练方向头 + 跑分侧方向门，返回门指标。

    `seed` 固定初始化（在构造 model **之前** `torch.manual_seed`）；落盘时另写
    `<out_path>.json` provenance sidecar，`.pt` 仍是裸 state_dict、格式不变。
    """
    encoder = DEFAULT_ENCODER
    tr_t, tr_y = _read_goemotions_signed(f"{data_dir}/train.tsv")
    te_t, te_y = _read_goemotions_signed(f"{data_dir}/test.tsv")
    if crowdenvent_path:
        ce_t, ce_y = _read_crowdenvent_signed(crowdenvent_path)
        logger.info("+ crowd-enVENT 叙事域 anger=%d fear=%d", sum(ce_y), len(ce_y) - sum(ce_y))
        tr_t = tr_t + ce_t
        tr_y = tr_y + ce_y
    n_anger = sum(tr_y)
    n_fear = len(tr_y) - n_anger
    logger.info(
        "train anger=%d fear=%d | test anger=%d fear=%d",
        n_anger,
        n_fear,
        sum(te_y),
        len(te_y) - sum(te_y),
    )

    x = encode_texts(tr_t, encoder=encoder)
    # 软标签(神经席首选)：anger 0.9 / fear 0.1；硬标签基线：1.0 / 0.0
    hi, lo = (1.0, 0.0) if hard else (0.9, 0.1)
    y = torch.tensor([hi if v == 1 else lo for v in tr_y], dtype=torch.float32)
    pos_weight = torch.tensor([n_fear / max(n_anger, 1)], dtype=torch.float32)  # 平衡多数 anger

    torch.manual_seed(seed)
    model = DirectionHead(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    final_loss = 0.0
    epochs_ran = 0
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if epoch % 100 == 0:
            logger.info("epoch %d BCE loss %.5f", epoch, final_loss)
    model.eval()

    # 分侧方向门：anger 侧 logit>0 正确、fear 侧 logit<0 正确
    with torch.no_grad():
        te_x = encode_texts(te_t, encoder=encoder)
        logit = model(te_x).tolist()
    anger_pred = [logit[i] for i in range(len(te_y)) if te_y[i] == 1]
    fear_pred = [logit[i] for i in range(len(te_y)) if te_y[i] == 0]
    a_ok = sum(1 for v in anger_pred if v > 0)
    f_ok = sum(1 for v in fear_pred if v < 0)
    a_lb = _wilson_lb(a_ok, len(anger_pred))
    f_lb = _wilson_lb(f_ok, len(fear_pred))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    write_provenance(
        out_path,
        script="scripts/train_direction_head.py",
        model=model,
        model_config={"hidden": 64, "dim": int(x.shape[1]), "encoder": encoder},
        # hard 决定软/硬标签（0.9/0.1 vs 1.0/0.0）；crowdenvent 决定训练集是否混入叙事域。
        # 两者都改变监督信号本身，必须随权重落账。
        data_config={"hard": hard, "crowdenvent_path": crowdenvent_path},
        data_source=data_dir,
        n_samples=int(x.shape[0]),
        seed=seed,
        lr=lr,
        epochs_requested=epochs,
        epochs_ran=epochs_ran,
        final_train_loss=final_loss,
        # 本脚本自带留出 test + 分侧方向门：这里记的是真·泛化面指标，非训练集拟合度。
        val={
            "split": "goemotions-test-holdout",
            "metric": "分侧方向正确率 + Wilson 95% 下界（每侧需 n≥50 且下界≥0.70）",
            "anger_rate": a_ok / len(anger_pred),
            "anger_n": len(anger_pred),
            "anger_wilson_lb": a_lb,
            "fear_rate": f_ok / len(fear_pred),
            "fear_n": len(fear_pred),
            "fear_wilson_lb": f_lb,
        },
    )

    return {
        "n_anger_tr": n_anger,
        "n_fear_tr": n_fear,
        "anger_rate": a_ok / len(anger_pred),
        "anger_n": len(anger_pred),
        "anger_lb": a_lb,
        "fear_rate": f_ok / len(fear_pred),
        "fear_n": len(fear_pred),
        "fear_lb": f_lb,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="符号监督 motivational_direction_prior 头 + 分侧门")
    p.add_argument("--data-dir", default="data/raw/goemotions")
    p.add_argument("--out", default="artifacts/motivational_direction_prior.pt")
    p.add_argument("--hard", action="store_true", help="硬标签(1/0)基线；默认软标签(0.9/0.1)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--crowdenvent",
        default=None,
        help="可选：crowd-enVent_generation.tsv 路径，混入叙事域 anger/fear 治 anger 跨源掉分",
    )
    p.add_argument("--seed", type=int, default=0, help="固定初始化，保证可复现")
    args = p.parse_args()

    m = train_and_gate(
        args.data_dir,
        args.out,
        hard=args.hard,
        epochs=args.epochs,
        lr=args.lr,
        crowdenvent_path=args.crowdenvent,
        seed=args.seed,
    )
    logger.info("─" * 62)
    logger.info("分侧方向门（GoEmotions test 留出；每侧需 n≥50 且 Wilson 下界≥0.70）:")
    logger.info(
        "  anger 侧 %d/%d=%.1f%%  Wilson 下界 %.4f",
        round(m["anger_rate"] * m["anger_n"]),
        m["anger_n"],
        m["anger_rate"] * 100,
        m["anger_lb"],
    )
    logger.info(
        "  fear  侧 %d/%d=%.1f%%  Wilson 下界 %.4f",
        round(m["fear_rate"] * m["fear_n"]),
        m["fear_n"],
        m["fear_rate"] * 100,
        m["fear_lb"],
    )
    passed = (
        m["anger_n"] >= 50 and m["fear_n"] >= 50 and m["anger_lb"] >= 0.70 and m["fear_lb"] >= 0.70
    )
    verdict = "PASS(两侧下界≥0.70)" if passed else "FAIL(某侧下界<0.70 或 n<50)"
    logger.info("─" * 62)
    logger.info("GoEmotions 单飞方向门：%s（标签模式=%s）", verdict, "硬" if args.hard else "软")
    print(
        f"DIR-GATE goemotions: anger_lb={m['anger_lb']:.3f} fear_lb={m['fear_lb']:.3f} => {verdict}"
    )


if __name__ == "__main__":
    main()
