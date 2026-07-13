"""在 WESAD 上训练 PhysiologyDecoder（(v,a)→真实生理特征），保存权重。

用法：python -m scripts.train_physiology --root data/wesad --epochs 300
数据获取见 src/agents/datasets/wesad.py 模块文档。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=16, num_layers=1）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.wesad import load_wesad
from src.agents.models.physiology_decoder import PhysiologyDecoder

logger = logging.getLogger(__name__)


def train(
    root: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    window_seconds: int = 30,
    limit: int | None = None,
    hidden: int = 16,
    num_layers: int = 1,
    out: str = "artifacts/physiology_decoder.pt",
) -> float:
    """加载 WESAD、全批量训练 PhysiologyDecoder 并保存权重，返回最终 MSE。"""
    x, y = load_wesad(root, window_seconds=window_seconds, limit=limit)
    model = PhysiologyDecoder(hidden=hidden, num_layers=num_layers)
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
    logger.info("saved physiology decoder -> %s (final loss %.6f)", out_path, final_loss)
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="WESAD 解压根目录（含 Sxx/Sxx.pkl）")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out", default="artifacts/physiology_decoder.pt")
    args = parser.parse_args()
    final = train(
        args.root,
        epochs=args.epochs,
        window_seconds=args.window_seconds,
        limit=args.limit,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=args.out,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
