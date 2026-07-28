"""在 RAVDESS 上训练 ProsodyDecoder（(v,a)→真实韵律），保存权重。

用法：python -m scripts.train_prosody --root data/ravdess --epochs 300
数据获取见 src/agents/datasets/ravdess.py 模块文档。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=16, num_layers=1）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from scripts._train_common import (
    DEFAULT_MAX_EPOCHS,
    add_stop_arguments,
    add_val_split_arguments,
    evaluate_with_baselines,
    resolve_cli_epochs,
    resolve_epoch_budget,
    split_by_groups,
    write_provenance,
)
from src.agents.datasets.ravdess import load_ravdess
from src.agents.models.prosody_decoder import ProsodyDecoder

logger = logging.getLogger(__name__)


def train(
    root: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    limit: int | None = None,
    hidden: int = 16,
    num_layers: int = 1,
    out: str = "artifacts/prosody_decoder.pt",
    stop: str = "plateau",
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_split: str = "none",
    val_seed: int = 0,
    seed: int = 0,
) -> float:
    """加载 RAVDESS、全批量训练 ProsodyDecoder 并保存权重，返回最终 MSE。

    `stop="plateau"`（默认）：训练 loss 每 100 步相对下降 <1e-4 即停、`max_epochs` 封顶，
    不再依赖 `epochs=300` 这个魔数（换个 lr 它就静默失效）。`stop="fixed"` 跑满 `epochs`，
    供既有调用方保持逐字旧行为。判据只看训练 loss，**不是**泛化最优点。
    `seed` 固定初始化，保证可复现；落盘时另写 `<out>.json` provenance sidecar
    （轮数/lr/种子/数据/commit），`.pt` 仍是裸 state_dict、格式不变。
    """
    x, y = load_ravdess(root, limit=limit)

    x_val = y_val = None
    split_info: dict[str, object] = {}
    if val_split != "none":
        # 分组键 = (v,a) 锚点。⚠ 这**挡不住说话人泄漏**——RAVDESS 的方差 61% 来自说话人身份，
        # 同一锚点的不同 actor 仍会分到两侧。要隔离说话人需按 actor 分组（另需 loader 返回
        # 说话人 id，同 P1-3 给 WESAD 做的那样）。
        groups = [tuple(row.tolist()) for row in x]
        x, y, x_val, y_val, val_groups, split_info = split_by_groups(
            x, y, groups, val_seed=val_seed
        )
        logger.info(
            "留出切分：train %d 行/%d 组 · val %d 行/%d 组",
            split_info["n_train_samples"],
            split_info["n_train_groups"],
            split_info["n_samples"],
            split_info["n_val_groups"],
        )

    torch.manual_seed(seed)
    model = ProsodyDecoder(hidden=hidden, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    budget, stopper = resolve_epoch_budget(stop=stop, epochs=epochs, max_epochs=max_epochs)
    model.train()
    final_loss = 0.0
    epochs_ran = 0
    for epoch in range(budget):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if epoch % 50 == 0:
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, x.shape[0])
        if stopper is not None and stopper.should_stop(epochs_ran, final_loss):
            logger.info("训练 loss 进入平台，停于 epoch %d（loss %.6f）", epochs_ran, final_loss)
            break

    val = None
    if x_val is not None:
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
        model.train()
        metrics = evaluate_with_baselines(
            y_val, pred_val, val_groups, train_mean=y.mean(dim=0, keepdim=True)
        )
        logger.info(
            "留出评估：技能分 %.4f（MSE %.6f · 常数基线 %.6f · 下界 %.6f · 可学空间走了 %.1f%%）",
            metrics["skill_score"],
            metrics["mse"],
            metrics["mse_constant"],
            metrics["within_group_floor"],
            metrics["headroom_used"] * 100,
        )
        val = {
            "split": f"group-holdout:{val_split}",
            "group_key": "(valence, arousal) 锚点",
            "note": "⚠ 按锚点分组，不隔离说话人；RAVDESS 方差 61% 来自说话人身份，此 val 偏乐观。",
            **metrics,
            **split_info,
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved prosody decoder -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_prosody.py",
        model=model,
        model_config={"hidden": hidden, "num_layers": num_layers},
        data_config={"limit": limit, "stop": stop, "val_split": val_split},
        data_source=root,
        n_samples=int(x.shape[0]),
        seed=seed,
        lr=lr,
        epochs_requested=budget,
        epochs_ran=epochs_ran,
        final_train_loss=final_loss,
        val=val,
    )
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="RAVDESS 解压根目录（含 Actor_xx/*.wav）")
    add_stop_arguments(parser)
    add_val_split_arguments(parser, choices=["none", "class"], group_key="(v,a) 锚点")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out", default="artifacts/prosody_decoder.pt")
    parser.add_argument("--seed", type=int, default=0, help="固定初始化，保证可复现")
    args = parser.parse_args()
    final = train(
        args.root,
        epochs=resolve_cli_epochs(args),
        stop=args.stop,
        max_epochs=args.max_epochs,
        val_split=args.val_split,
        val_seed=args.val_seed,
        limit=args.limit,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=args.out,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
