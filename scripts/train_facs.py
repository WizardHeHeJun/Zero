"""在 FACS 标注 CSV 上训练 FacsDecoder（(v,a)→AU 强度），保存权重。

用法（旧 5-AU）：python -m scripts.train_facs --csv data/facs/labels.csv --epochs 300
用法（13-AU 扩展）：python -m scripts.train_facs --csv data/facs/labels_ext.csv --ext \
    --out artifacts/facs_decoder_ext_v2.pt

数据获取/CSV 格式见 src/agents/datasets/facs_csv.py 模块文档。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=16, num_layers=1）。

--ext 模式说明：
  输出文件默认 facs_decoder_ext.pt；13-AU v2 建议显式传 --out artifacts/facs_decoder_ext_v2.pt，
  与旧 5-AU facs_decoder.pt 隔离。当前已有 laion/emonet-face-binary（CC-BY-4.0）
  + OpenFace AU 的可跑训练路径；AffectNet/DISFA 仍可作为后续更高保真逐帧 AU 数据源。
  FacsDecoder 仍是 predict_facs(v, a)，不吃 coping_potential；愤怒/恐惧 coping 分野由
  CompositeChannelDecoder residual / 解析占位承担，真权重主要学习通用 AU 真实度。
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
from src.agents.datasets.facs_csv import load_facs_csv, load_facs_csv_ext
from src.agents.models.composite import _COPING_DRIVEN_AUS
from src.agents.models.facs_decoder import FACS_KEYS_EXT, FacsDecoder

logger = logging.getLogger(__name__)

# 运行时真正消费的输出维。默认 residual_alpha=1.0 时 composite.py 把 coping 判别 AU
# 整个换成解析占位（`base*(1-α) + placeholder*α`），真模型这 4 维的输出从不被读取——
# 把它们算进评估指标是自欺。两个常量直接引用而非复制，避免任一侧改动后指标悄悄失真。
_RUNTIME_CONSUMED_COLUMNS = [
    i for i, key in enumerate(FACS_KEYS_EXT) if key not in _COPING_DRIVEN_AUS
]


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    hidden: int = 16,
    num_layers: int = 1,
    out: str = "artifacts/facs_decoder.pt",
    extended: bool = False,
    stop: str = "plateau",
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_split: str = "none",
    val_seed: int = 0,
    seed: int = 0,
) -> float:
    """加载 FACS CSV、全批量训练 FacsDecoder 并保存权重，返回最终 MSE。

    extended=True：加载 13-AU 扩展 CSV（load_facs_csv_ext），用 FacsDecoder(extended=True)。
    现有 EmoNet/OpenFace 路径已可训练；AffectNet/DISFA 仍是后续更高保真数据源。
    13-AU v2 建议输出到 facs_decoder_ext_v2.pt，与旧 5-AU / 11-AU 权重隔离。

    `stop="plateau"`（默认）：训练 loss 每 100 步相对下降 <1e-4 即停、`max_epochs` 封顶，
    不再依赖 `epochs=300` 这个魔数（换个 lr 它就静默失效）。`stop="fixed"` 跑满 `epochs`，
    供既有调用方保持逐字旧行为。判据只看训练 loss，**不是**泛化最优点。
    `seed` 固定初始化，保证可复现；落盘时另写 `<out>.json` provenance sidecar
    （轮数/lr/种子/数据/commit），`.pt` 仍是裸 state_dict、格式不变。
    """
    if extended:
        x, y = load_facs_csv_ext(csv_path)
    else:
        x, y = load_facs_csv(csv_path)

    x_val = y_val = None
    split_info: dict[str, object] = {}
    if val_split != "none":
        # 分组键 = 每行的 (v,a) 锚点：同一锚点常有几十行，按行随机切会让它们跨切分泄漏，
        # val 退化成「锚点均值下界」，给出「永远训更久更好」的假信号。
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
    model = FacsDecoder(hidden=hidden, num_layers=num_layers, extended=extended)
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
            y_val,
            pred_val,
            val_groups,
            train_mean=y.mean(dim=0, keepdim=True),  # 训练集均值——用 val 的均值就是泄漏
            columns=_RUNTIME_CONSUMED_COLUMNS if extended else None,
        )
        logger.info(
            "留出评估：技能分 %.4f（MSE %.6f · 常数基线 %.6f · 下界 %.6f · 可学空间走了 %.1f%%）",
            metrics["skill_score"],
            metrics["mse"],
            metrics["mse_constant"],
            metrics["within_group_floor"],
            metrics["headroom_used"] * 100,
        )
        scored_dims = (
            "运行时消费的 9 维（剔除 composite 默认 α=1.0 下被占位覆盖的 AU23/01/02/20）"
            if extended
            else "全部输出维"
        )
        val = {
            "split": f"group-holdout:{val_split}",
            "group_key": "(valence, arousal) 锚点",
            "scored_dims": scored_dims,
            "note": "val 里的锚点在 train 中一次都没出现过；模型未在 val 上做过任何选择。",
            **metrics,
            **split_info,
        }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved facs decoder -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_facs.py",
        model=model,
        model_config={"hidden": hidden, "num_layers": num_layers, "extended": extended},
        data_config={"stop": stop, "val_split": val_split},
        data_source=csv_path,
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
    parser.add_argument("--csv", required=True, help="FACS 标注 CSV 路径")
    add_stop_arguments(parser)
    add_val_split_arguments(parser, choices=["none", "class"], group_key="(v,a) 锚点")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--ext",
        action="store_true",
        default=False,
        help="训练 13-AU 扩展模型（emonet CC-BY + OpenFace 路径已可训；建议 --out ..._ext_v2.pt）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "权重输出路径"
            "（默认：--ext→artifacts/facs_decoder_ext.pt；否则→artifacts/facs_decoder.pt）"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="固定初始化，保证可复现")
    args = parser.parse_args()
    default_out = "artifacts/facs_decoder_ext.pt" if args.ext else "artifacts/facs_decoder.pt"
    out_path = args.out if args.out is not None else default_out
    final = train(
        args.csv,
        epochs=resolve_cli_epochs(args),
        stop=args.stop,
        max_epochs=args.max_epochs,
        val_split=args.val_split,
        val_seed=args.val_seed,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=out_path,
        extended=args.ext,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
