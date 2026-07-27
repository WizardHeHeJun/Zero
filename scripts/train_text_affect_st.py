"""在 EmoBank 上训练 STTextAffectRegressor（句向量→(v,a)），保存权重。

相比 train_text_affect.py 的哈希词袋，前端换成**冻结**的预训练句向量编码器，只训上面的
MLP 头——带语义泛化、跨域/未见词更稳。句向量编码一次性预计算（最慢的一步）。
用法：python -m scripts.train_text_affect_st --csv data/emobank.csv --epochs 300

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=64, num_layers=1）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from src.agents.datasets.emobank import read_emobank_rows
from src.agents.datasets.emobank_st import load_emobank_embeddings
from src.agents.models.text_affect_regressor_st import (
    DEFAULT_ENCODER,
    ST_FEATURE_DIM,
    STTextAffectRegressor,
)

logger = logging.getLogger(__name__)


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    encoder: str = DEFAULT_ENCODER,
    hidden: int = 64,
    num_layers: int = 1,
    finetune_encoder: bool = False,
    out: str = "artifacts/text_affect_regressor_st.pt",
    seed: int = 0,
) -> float:
    """加载 EmoBank→句向量、全批量训练 STTextAffectRegressor 并保存权重，返回最终 MSE。

    `seed` 固定初始化，保证可复现。
    """
    if finetune_encoder:
        # 端到端微调（W4 修复）：喂原始文本，前向经 encoder_module（带梯度），编码器参数参与
        # backward（需 GPU；forward(预计算张量) 路径编码器不在计算图内、拿不到梯度）。
        texts, ys = read_emobank_rows(csv_path, limit=limit)
        y = torch.tensor(ys, dtype=torch.float32)
        torch.manual_seed(seed)
        model = STTextAffectRegressor(
            dim=ST_FEATURE_DIM,
            hidden=hidden,
            num_layers=num_layers,
            encoder=encoder,
            finetune_encoder=True,
        )
        n = len(texts)

        def _predict() -> torch.Tensor:
            return model.forward_texts(texts)
    else:
        logger.info("encoding texts with %s (one-time, may take ~1min on CPU)...", encoder)
        x, y = load_emobank_embeddings(csv_path, encoder=encoder, limit=limit)
        torch.manual_seed(seed)
        model = STTextAffectRegressor(
            dim=x.shape[1],
            hidden=hidden,
            num_layers=num_layers,
            encoder=encoder,
            finetune_encoder=False,
        )
        n = x.shape[0]

        def _predict() -> torch.Tensor:
            return model(x)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(_predict(), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 50 == 0:
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, n)

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
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--finetune-encoder",
        action="store_true",
        default=False,
        help="端到端微调句向量编码器（需 GPU，MiniLM ~22M CPU 极慢）",
    )
    parser.add_argument("--out", default="artifacts/text_affect_regressor_st.pt")
    parser.add_argument("--seed", type=int, default=0, help="固定初始化，保证可复现")
    args = parser.parse_args()
    final = train(
        args.csv,
        epochs=args.epochs,
        limit=args.limit,
        encoder=args.encoder,
        hidden=args.hidden,
        num_layers=args.num_layers,
        finetune_encoder=args.finetune_encoder,
        out=args.out,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
