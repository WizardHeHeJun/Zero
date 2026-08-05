"""在 RAVDESS 头姿轨迹上训练 MotionDecoder（(v,a)→运动学调制系数），保存权重。

用法：python -m scripts.train_motion --root data/ravdess_motion --val-split actor
数据获取见 src/agents/datasets/ravdess_motion.py 模块文档。

⚠ **本通道的验收要求同时报两组留出**（议会数学席定为必测项）：
  --val-split anchor  按 (v,a) 锚点分组
  --val-split actor   按演员分组（leave-actor-out）
若 anchor 组技能分明显为正、而 actor 组塌到 0 附近甚至为负，**即是说话人泄漏的直接
数学信号**——此时不得发布权重。同源教训：韵律通道实测 61% 方差来自说话人身份，
且当年「标注了局限但没做隔离」。本通道实测演员间方差 27–28%（幅度/速度），
比韵律好但仍是主导项，故隔离不可省。

量纲：loader 返回**原始量纲**统计量，三维尺度差约 4 个数量级
（amplitude≈0.06 · speed≈0.29 · max_jerk≈1244）。不归一化则 MSE 完全被 jerk 主导、
另两维等于没训。故此处按**训练集**逐维标准化，并把均值/标准差写进 provenance
——推理侧据此反变换，两处口径同源。
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
from src.agents.datasets.ravdess_motion import MOTION_STAT_KEYS, load_ravdess_motion
from src.agents.models.motion_decoder import AnchorInterpolator, MotionMLP

logger = logging.getLogger(__name__)


def train(
    root: str,
    *,
    epochs: int = 300,
    lr: float = 1e-2,
    limit: int | None = None,
    arch: str = "interp",
    hidden: int = 8,
    out: str = "artifacts/motion_decoder.pt",
    stop: str = "plateau",
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_split: str = "none",
    val_seed: int = 0,
    seed: int = 0,
) -> float:
    """加载头姿数据、全批量训练调制模型并保存权重，返回最终（标准化尺度上的）MSE。

    `arch="interp"`（默认）用 `AnchorInterpolator`（自由度=锚点数，规避 15 锚点下的
    结构性不可辨识）；`arch="mlp"` 用小 MLP 作消融对照。
    """
    x, y, groups = load_ravdess_motion(root, limit=limit, return_groups=True)
    logger.info(
        "载入 %d 条轨迹 · %d 演员 · %d 个 (v,a) 锚点",
        x.shape[0],
        len(set(groups)),
        len({tuple(r.tolist()) for r in x}),
    )

    x_val = y_val = None
    val_groups: list[str] = []
    split_info: dict[str, object] = {}
    if val_split != "none":
        # ⚠ 分组键决定这个 val 测的是什么：
        #   anchor —— 同一锚点整组进一侧，但**同锚点的不同演员仍会跨切分** ⇒ 偏乐观；
        #   actor  —— 同一演员整组进一侧，才真正隔离个体差异（本通道 27–28% 方差在此）。
        keys: list[object]
        if val_split == "actor":
            keys = list(groups)
        else:
            keys = [tuple(row.tolist()) for row in x]
        x, y, x_val, y_val, val_groups, split_info = split_by_groups(x, y, keys, val_seed=val_seed)
        logger.info(
            "留出切分(%s)：train %d 行/%d 组 · val %d 行/%d 组",
            val_split,
            split_info["n_train_samples"],
            split_info["n_train_groups"],
            split_info["n_samples"],
            split_info["n_val_groups"],
        )

    # 逐维标准化（**用训练集统计量**，防 val 信息回流）。不做的话 loss 只等于在训 jerk。
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True).clamp(min=1e-8)
    y_norm = (y - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std if y_val is not None else None

    torch.manual_seed(seed)
    anchors = torch.unique(x, dim=0)
    model: nn.Module = (
        AnchorInterpolator(anchors, out_dim=y.shape[1])
        if arch == "interp"
        else MotionMLP(out_dim=y.shape[1], hidden=hidden)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    budget, stopper = resolve_epoch_budget(stop=stop, epochs=epochs, max_epochs=max_epochs)
    model.train()
    final_loss = 0.0
    epochs_ran = 0
    for epoch in range(budget):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y_norm)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if epoch % 100 == 0:
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, x.shape[0])
        if stopper is not None and stopper.should_stop(epochs_ran, final_loss):
            logger.info("训练 loss 进入平台，停于 epoch %d（loss %.6f）", epochs_ran, final_loss)
            break

    val = None
    if x_val is not None and y_val_norm is not None:
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
        model.train()
        metrics = evaluate_with_baselines(
            y_val_norm, pred_val, val_groups, train_mean=y_norm.mean(dim=0, keepdim=True)
        )
        logger.info(
            "留出评估(%s)：技能分 %.4f（MSE %.6f · 常数基线 %.6f · "
            "下界 %.6f · 可学空间走了 %.1f%%）",
            val_split,
            metrics["skill_score"],
            metrics["mse"],
            metrics["mse_constant"],
            metrics["within_group_floor"],
            metrics["headroom_used"] * 100,
        )
        note = (
            "按演员分组（leave-actor-out）：真正隔离个体差异，本通道 27–28% 方差在此。"
            if val_split == "actor"
            else "⚠ 按锚点分组，**不隔离演员**；同锚点不同演员会跨切分 ⇒ 此 val 偏乐观。"
        )
        val = {
            "split": f"group-holdout:{val_split}",
            "group_key": "演员 id" if val_split == "actor" else "(valence, arousal) 锚点",
            "note": note,
            "scale": "标准化尺度（逐维 z-score，统计量见 target_norm）",
            **metrics,
            **split_info,
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved motion decoder -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_motion.py",
        model=model,
        model_config={
            "arch": arch,
            "hidden": hidden if arch == "mlp" else None,
            "n_anchors": int(anchors.shape[0]),
            "stat_keys": list(MOTION_STAT_KEYS),
            # 推理侧反标准化必需；与训练同源，不许两处各写一份
            "target_norm": {
                "mean": [round(v, 8) for v in y_mean.flatten().tolist()],
                "std": [round(v, 8) for v in y_std.flatten().tolist()],
            },
        },
        data_config={
            "limit": limit,
            "stop": stop,
            "val_split": val_split,
            "domain_note": "posed（演员摆拍）训练，未做 acted→spontaneous 域适配；"
            "幅度斜率上界偏陡，运行时须经 ZERO_MOTION_AMPLITUDE_SCALE 衰减",
        },
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
    parser.add_argument("--root", required=True, help="解压根目录（含 *.csv）")
    add_stop_arguments(parser)
    add_val_split_arguments(
        parser, choices=["none", "anchor", "actor"], group_key="(v,a) 锚点 或 演员 id"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arch", choices=["interp", "mlp"], default="interp")
    parser.add_argument("--hidden", type=int, default=8, help="仅 --arch mlp 生效")
    parser.add_argument("--out", default="artifacts/motion_decoder.pt")
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
        arch=args.arch,
        hidden=args.hidden,
        out=args.out,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
