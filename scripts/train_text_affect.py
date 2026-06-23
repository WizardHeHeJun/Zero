"""在 EmoBank 上训练 TextAffectRegressor（文本→(v,a)），保存权重。

用法：python -m scripts.train_text_affect --csv data/emobank.csv --epochs 300
数据获取/格式见 src/agents/datasets/emobank.py 模块文档。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.emobank import load_emobank
from src.agents.models.text_affect_regressor import TextAffectRegressor

logger = logging.getLogger(__name__)


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    out: str = "artifacts/text_affect_regressor.pt",
) -> float:
    """加载 EmoBank、全批量训练 TextAffectRegressor 并保存权重，返回最终 MSE。"""
    x, y = load_emobank(csv_path, limit=limit)
    model = TextAffectRegressor()
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
    logger.info("saved text affect regressor -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="EmoBank CSV 路径")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/text_affect_regressor.pt")
    args = parser.parse_args()
    final = train(args.csv, epochs=args.epochs, limit=args.limit, out=args.out)
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
