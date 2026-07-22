"""在 EmoBank 上训练 D 维（Dominance）或 VAD 全维句向量回归器，保存权重。

复用 STTextAffectRegressor + load_emobank_embeddings(include_d=True) 路径。
--target 控制训练目标维度：
  va  → output_dim=2，仅 V/A（等价于 train_text_affect_st.py 默认）
  d   → output_dim=1，仅 D（SAM Dominance）
  vad → output_dim=3，V/A/D 全维

阻塞说明（使用前须知）：
  1. D 语义对齐悬而未决：EmoBank D 列（SAM Dominance）与 coping_potential/control
     appraisal 的操作化对应尚待议会裁定（议会 2026-07-13 #2），D 单源于 EmoBank、
     无跨数据集 D 标注可交叉验证。
  2. 初版 coping_potential 不进多模态融合：D 维监督训练属于扩展特性，须议会排后、
     明确操作化定义后才挂入主融合通道。
  3. 韵律/生理数据集（RAVDESS/WESAD）无逐样本连续 D 标注；D 维监督单源 EmoBank，
     数学席已警告跨通道融合的方差估计不可靠。
用法：python -m scripts.train_text_affect_d --csv data/emobank.csv --target d --epochs 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.emobank_st import load_emobank_embeddings
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, STTextAffectRegressor

logger = logging.getLogger(__name__)

_TARGET_TO_DIM: dict[str, int] = {"va": 2, "d": 1, "vad": 3}


def train(
    csv_path: str,
    *,
    target: str = "d",
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    encoder: str = DEFAULT_ENCODER,
    hidden: int = 64,
    num_layers: int = 1,
    out: str = "artifacts/text_affect_regressor_d.pt",
) -> float:
    """加载 EmoBank→句向量（含 D）、训练 STTextAffectRegressor 并保存权重，返回最终 MSE。

    Args:
        csv_path: EmoBank CSV 路径（需含 V,A,D,text 列）。
        target: 训练目标，"va"/"d"/"vad"，对应 output_dim 2/1/3。
        epochs: 训练轮数。
        lr: Adam 学习率。
        limit: 最多使用样本数，None 表示全量。
        encoder: 句向量编码器名。
        hidden: MLP 隐层宽度。
        num_layers: MLP 隐层数。
        out: 权重输出路径。

    Returns:
        最终 MSE 损失值。
    """
    if target not in _TARGET_TO_DIM:
        raise ValueError(f"--target 须为 va/d/vad，得到 {target!r}")
    output_dim = _TARGET_TO_DIM[target]

    logger.info("encoding texts with %s (one-time, may take ~1min on CPU)...", encoder)
    x, y_full = load_emobank_embeddings(csv_path, encoder=encoder, limit=limit, include_d=True)
    # 按 target 切列
    if target == "va":
        y = y_full[:, :2]
    elif target == "d":
        y = y_full[:, 2:3]
    else:  # vad
        y = y_full

    model = STTextAffectRegressor(
        dim=x.shape[1],
        hidden=hidden,
        num_layers=num_layers,
        encoder=encoder,
        output_dim=output_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 50 == 0:
            logger.info(
                "epoch %d loss %.6f (n=%d, target=%s)", epoch, final_loss, x.shape[0], target
            )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info(
        "saved text affect D regressor -> %s (target=%s, final loss %.6f)",
        out_path,
        target,
        final_loss,
    )
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="训练 EmoBank D 维（Dominance）或 VAD 全维句向量回归器。"
    )
    parser.add_argument("--csv", required=True, help="EmoBank CSV 路径")
    parser.add_argument(
        "--target",
        choices=["va", "d", "vad"],
        default="d",
        help="训练目标：va=仅V/A / d=仅D / vad=V/A/D 全维（默认 d）",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out", default="artifacts/text_affect_regressor_d.pt")
    args = parser.parse_args()
    final = train(
        args.csv,
        target=args.target,
        epochs=args.epochs,
        limit=args.limit,
        encoder=args.encoder,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=args.out,
    )
    print(f"done, target={args.target}, final loss={final:.6f}")


if __name__ == "__main__":
    main()
