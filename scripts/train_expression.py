"""训练 ExpressionDecoder：把 (v,a)→通道 解析占位蒸馏成可训练网络（合成数据 bootstrap）。

用法：python -m scripts.train_expression --epochs 300 --out artifacts/expression_decoder.pt
真数据到位后，把 synthetic_pairs 换成真实 DataLoader 即可，本训练循环复用。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=32, num_layers=2）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.synthetic import synthetic_pairs
from src.agents.models.expression_decoder import ExpressionDecoder

logger = logging.getLogger(__name__)


def train(
    *,
    epochs: int = 300,
    n: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 32,
    num_layers: int = 2,
    out: str = "artifacts/expression_decoder.pt",
) -> float:
    """全批量训练解码器并保存权重，返回最终 MSE 损失。"""
    x, y = synthetic_pairs(n, seed=seed)
    model = ExpressionDecoder(hidden=hidden, num_layers=num_layers)
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
            logger.info("epoch %d loss %.6f", epoch, final_loss)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved decoder -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out", default="artifacts/expression_decoder.pt")
    args = parser.parse_args()
    final = train(
        epochs=args.epochs, n=args.n, hidden=args.hidden, num_layers=args.num_layers, out=args.out
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
