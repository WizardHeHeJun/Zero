"""在 FACS 标注 CSV 上训练 FacsDecoder（(v,a)→AU 强度），保存权重。

用法：python -m scripts.train_facs --csv data/facs/labels.csv --epochs 300
数据获取/CSV 格式见 src/agents/datasets/facs_csv.py 模块文档。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.facs_csv import load_facs_csv
from src.agents.models.facs_decoder import FacsDecoder

logger = logging.getLogger(__name__)


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    out: str = "artifacts/facs_decoder.pt",
) -> float:
    """加载 FACS CSV、全批量训练 FacsDecoder 并保存权重，返回最终 MSE。"""
    x, y = load_facs_csv(csv_path)
    model = FacsDecoder()
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
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, x.shape[0])

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved facs decoder -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="FACS 标注 CSV 路径")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", default="artifacts/facs_decoder.pt")
    args = parser.parse_args()
    final = train(args.csv, epochs=args.epochs, out=args.out)
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
