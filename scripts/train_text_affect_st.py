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

from scripts._train_common import write_provenance
from src.agents.datasets.emobank import read_emobank_rows
from src.agents.models.text_affect_regressor_st import (
    DEFAULT_ENCODER,
    ST_FEATURE_DIM,
    STTextAffectRegressor,
    encode_texts,
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
    official_split: bool = True,
    patience: int = 50,
    seed: int = 0,
) -> float:
    """加载 EmoBank→句向量、全批量训练 STTextAffectRegressor 并保存权重，返回最终 MSE。

    `official_split=True`（默认）：只用官方 train 训练，用 dev 做早停并**返回 dev MSE**——
    否则读全量会把 dev/test 训进去，报出的分数是记忆不是泛化（见 emobank 模块文档）。
    本脚本此前**没有**这个开关、一直读全量，`weights-v0.1` 的 `text_affect_regressor_st.pt`
    即产自那条路径，其「loss 0.0056」是训练集拟合度，与词袋版的 0.016 不可比。
    早停保留 dev 最优权重；`patience` 轮无改善即停。`seed` 固定初始化，保证可复现。
    `official_split=False` 走旧路径（全量训练、返回训练 loss），仅供无 split 列的数据用。
    落盘时另写 `<out>.json` provenance sidecar，`.pt` 仍是裸 state_dict、格式不变。
    """
    dev_texts: list[str] | None = None
    dev_ys: list[list[float]] | None = None
    if official_split:
        try:
            texts, ys = read_emobank_rows(csv_path, limit=limit, split="train")
            dev_texts, dev_ys = read_emobank_rows(csv_path, limit=limit, split="dev")
        except ValueError as exc:
            # 小夹具/无官方切分的 CSV：降级读全量（旧行为），但明确告警——此时报出的
            # loss 是训练集 loss，不可当泛化指标用。
            logger.warning("官方切分不可用（%s），降级为全量训练；返回的是训练 loss", exc)
            texts, ys = read_emobank_rows(csv_path, limit=limit)
            dev_texts, dev_ys = None, None
    else:
        texts, ys = read_emobank_rows(csv_path, limit=limit)

    y = torch.tensor(ys, dtype=torch.float32)
    y_dev = torch.tensor(dev_ys, dtype=torch.float32) if dev_ys is not None else None

    if finetune_encoder:
        # 端到端微调（W4 修复）：喂原始文本，前向经 encoder_module（带梯度），编码器参数参与
        # backward（需 GPU；forward(预计算张量) 路径编码器不在计算图内、拿不到梯度）。
        torch.manual_seed(seed)
        model = STTextAffectRegressor(
            dim=ST_FEATURE_DIM,
            hidden=hidden,
            num_layers=num_layers,
            encoder=encoder,
            finetune_encoder=True,
        )
        n = len(texts)
        feature_dim = ST_FEATURE_DIM

        def _predict() -> torch.Tensor:
            return model.forward_texts(texts)

        def _predict_dev() -> torch.Tensor:
            assert dev_texts is not None
            return model.forward_texts(dev_texts)
    else:
        logger.info("encoding texts with %s (one-time, may take ~1min on CPU)...", encoder)
        x = encode_texts(texts, encoder=encoder)
        x_dev = encode_texts(dev_texts, encoder=encoder) if dev_texts is not None else None
        torch.manual_seed(seed)
        model = STTextAffectRegressor(
            dim=x.shape[1],
            hidden=hidden,
            num_layers=num_layers,
            encoder=encoder,
            finetune_encoder=False,
        )
        n = x.shape[0]
        feature_dim = int(x.shape[1])

        def _predict() -> torch.Tensor:
            return model(x)

        def _predict_dev() -> torch.Tensor:
            return model(x_dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0.0
    epochs_ran = 0
    best_dev, best_state, stale = float("inf"), None, 0
    best_epoch, train_loss_at_best = -1, float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(_predict(), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if y_dev is not None:
            model.eval()
            with torch.no_grad():
                dev = float(loss_fn(_predict_dev(), y_dev).item())
            model.train()
            if dev < best_dev - 1e-6:
                best_dev, stale = dev, 0
                best_epoch, train_loss_at_best = epoch, final_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch % 50 == 0:
                logger.info("epoch %d train %.6f dev %.6f (n=%d)", epoch, final_loss, dev, n)
            if stale >= patience:
                logger.info("dev 连续 %d 轮无改善，早停于 epoch %d", patience, epoch)
                break
        elif epoch % 50 == 0:
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, n)

    last_train_loss = final_loss  # 早停会把 final_loss 改写成 dev 指标，先留住训练 loss 本身
    if best_state is not None:
        model.load_state_dict(best_state)  # 恢复 dev 最优，而非最后一轮
        final_loss = best_dev
        logger.info("恢复 dev 最优权重（dev MSE %.6f）", best_dev)

    val = None
    if y_dev is not None:
        val = {
            "split": "emobank-official-dev",
            "mse": best_dev,
            "n_samples": int(y_dev.shape[0]),
            "patience": patience,
            "best_epoch": best_epoch,
            "train_loss_at_best": train_loss_at_best,
            "note": "保存的是 dev 最优轮权重（非末轮）；train() 返回值即此 mse。",
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved ST text affect regressor -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_text_affect_st.py",
        model=model,
        model_config={
            "hidden": hidden,
            "num_layers": num_layers,
            "encoder": encoder,
            "dim": feature_dim,
            "finetune_encoder": finetune_encoder,
        },
        data_config={
            "limit": limit,
            "official_split_requested": official_split,
            "official_split_used": y_dev is not None,
        },
        data_source=csv_path,
        n_samples=int(n),
        seed=seed,
        lr=lr,
        epochs_requested=epochs,
        epochs_ran=epochs_ran,
        final_train_loss=last_train_loss,
        val=val,
    )
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
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="旧路径：读全量（含官方 dev/test）训练。⚠ 会污染评测，仅供无 split 列的数据",
    )
    parser.add_argument("--patience", type=int, default=50, help="dev 无改善多少轮后早停")
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
        official_split=not args.full_data,
        patience=args.patience,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
