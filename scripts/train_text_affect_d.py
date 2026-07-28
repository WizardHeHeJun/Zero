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

from scripts._train_common import write_provenance
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
    official_split: bool = True,
    patience: int = 50,
    seed: int = 0,
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
        official_split: True（默认）只用官方 train 训练、用 dev 早停并**返回 dev MSE**；
            False 走旧路径读全量（会把 dev/test 训进去，报出的分数是记忆不是泛化）。
            本脚本此前没有这个开关、一直读全量——D 维本就单源 EmoBank、无跨数据集可交叉
            验证，再把 dev/test 训进去等于完全没有独立评估面。
        patience: dev 连续多少轮无改善即早停。
        seed: 固定初始化，保证可复现。

    Returns:
        最终 MSE 损失值（`official_split=True` 时为 dev MSE）。

    落盘时另写 `<out>.json` provenance sidecar（轮数/lr/种子/target/编码器/数据/commit），
    `.pt` 仍是裸 state_dict、格式不变。
    """
    if target not in _TARGET_TO_DIM:
        raise ValueError(f"--target 须为 va/d/vad，得到 {target!r}")
    output_dim = _TARGET_TO_DIM[target]

    def _slice(full: torch.Tensor) -> torch.Tensor:
        """按 target 切列：va→前两列 / d→第三列 / vad→全三列。"""
        if target == "va":
            return full[:, :2]
        if target == "d":
            return full[:, 2:3]
        return full

    logger.info("encoding texts with %s (one-time, may take ~1min on CPU)...", encoder)
    x_dev: torch.Tensor | None = None
    y_dev: torch.Tensor | None = None
    if official_split:
        try:
            x, y_full = load_emobank_embeddings(
                csv_path, encoder=encoder, limit=limit, include_d=True, split="train"
            )
            x_dev, y_dev_full = load_emobank_embeddings(
                csv_path, encoder=encoder, limit=limit, include_d=True, split="dev"
            )
            y_dev = _slice(y_dev_full)
        except ValueError as exc:
            # 小夹具/无官方切分的 CSV：降级读全量（旧行为），但明确告警——此时报出的
            # loss 是训练集 loss，不可当泛化指标用。
            logger.warning("官方切分不可用（%s），降级为全量训练；返回的是训练 loss", exc)
            x, y_full = load_emobank_embeddings(
                csv_path, encoder=encoder, limit=limit, include_d=True
            )
            x_dev, y_dev = None, None
    else:
        x, y_full = load_emobank_embeddings(csv_path, encoder=encoder, limit=limit, include_d=True)
    y = _slice(y_full)

    torch.manual_seed(seed)
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
    epochs_ran = 0
    best_dev, best_state, stale = float("inf"), None, 0
    best_epoch, train_loss_at_best = -1, float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if x_dev is not None and y_dev is not None:
            model.eval()
            with torch.no_grad():
                dev = float(loss_fn(model(x_dev), y_dev).item())
            model.train()
            if dev < best_dev - 1e-6:
                best_dev, stale = dev, 0
                best_epoch, train_loss_at_best = epoch, final_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch % 50 == 0:
                logger.info(
                    "epoch %d train %.6f dev %.6f (n=%d, target=%s)",
                    epoch,
                    final_loss,
                    dev,
                    x.shape[0],
                    target,
                )
            if stale >= patience:
                logger.info("dev 连续 %d 轮无改善，早停于 epoch %d", patience, epoch)
                break
        elif epoch % 50 == 0:
            logger.info(
                "epoch %d loss %.6f (n=%d, target=%s)", epoch, final_loss, x.shape[0], target
            )

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
    write_provenance(
        out_path,
        script="scripts/train_text_affect_d.py",
        model=model,
        model_config={
            "hidden": hidden,
            "num_layers": num_layers,
            "encoder": encoder,
            "dim": int(x.shape[1]),
            "output_dim": output_dim,
        },
        data_config={
            "limit": limit,
            "target": target,
            "official_split_requested": official_split,
            "official_split_used": y_dev is not None,
        },
        data_source=csv_path,
        n_samples=int(x.shape[0]),
        seed=seed,
        lr=lr,
        epochs_requested=epochs,
        epochs_ran=epochs_ran,
        final_train_loss=last_train_loss,
        val=val,
    )
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
        target=args.target,
        epochs=args.epochs,
        limit=args.limit,
        encoder=args.encoder,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=args.out,
        official_split=not args.full_data,
        patience=args.patience,
        seed=args.seed,
    )
    print(f"done, target={args.target}, final loss={final:.6f}")


if __name__ == "__main__":
    main()
