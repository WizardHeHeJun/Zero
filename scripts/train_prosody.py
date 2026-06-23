"""在 RAVDESS 上训练 ProsodyDecoder（(v,a)→真实韵律），保存权重。

用法：python -m scripts.train_prosody --root data/ravdess --epochs 300
数据获取见 src/agents/datasets/ravdess.py 模块文档。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.ravdess import load_ravdess
from src.agents.models.prosody_decoder import ProsodyDecoder

logger = logging.getLogger(__name__)


def train(
    root: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    out: str = "artifacts/prosody_decoder.pt",
) -> float:
    """加载 RAVDESS、全批量训练 ProsodyDecoder 并保存权重，返回最终 MSE。"""
    x, y = load_ravdess(root, limit=limit)
    model = ProsodyDecoder()
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
    logger.info("saved prosody decoder -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="RAVDESS 解压根目录（含 Actor_xx/*.wav）")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/prosody_decoder.pt")
    args = parser.parse_args()
    final = train(args.root, epochs=args.epochs, limit=args.limit, out=args.out)
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
