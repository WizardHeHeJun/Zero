"""在 EmoBank 上训练 STTextAffectRegressor（句向量→(v,a)），保存权重。

相比 train_text_affect.py 的哈希词袋，前端换成**冻结**的预训练句向量编码器，只训上面的
MLP 头——带语义泛化、跨域/未见词更稳。句向量编码一次性预计算（最慢的一步）。
用法：python -m scripts.train_text_affect_st --csv data/emobank.csv --epochs 300
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


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    encoder: str = DEFAULT_ENCODER,
    out: str = "artifacts/text_affect_regressor_st.pt",
) -> float:
    """加载 EmoBank→句向量、全批量训练 STTextAffectRegressor 并保存权重，返回最终 MSE。"""
    logger.info("encoding texts with %s (one-time, may take ~1min on CPU)...", encoder)
    x, y = load_emobank_embeddings(csv_path, encoder=encoder, limit=limit)
    model = STTextAffectRegressor(dim=x.shape[1], encoder=encoder)
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
    logger.info("saved ST text affect regressor -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="EmoBank CSV 路径")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--out", default="artifacts/text_affect_regressor_st.pt")
    args = parser.parse_args()
    final = train(
        args.csv, epochs=args.epochs, limit=args.limit, encoder=args.encoder, out=args.out
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
