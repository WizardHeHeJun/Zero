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

from scripts._train_common import write_provenance
from src.agents.datasets.facs_csv import load_facs_csv, load_facs_csv_ext
from src.agents.models.facs_decoder import FacsDecoder

logger = logging.getLogger(__name__)


def train(
    csv_path: str,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    hidden: int = 16,
    num_layers: int = 1,
    out: str = "artifacts/facs_decoder.pt",
    extended: bool = False,
    seed: int = 0,
) -> float:
    """加载 FACS CSV、全批量训练 FacsDecoder 并保存权重，返回最终 MSE。

    extended=True：加载 13-AU 扩展 CSV（load_facs_csv_ext），用 FacsDecoder(extended=True)。
    现有 EmoNet/OpenFace 路径已可训练；AffectNet/DISFA 仍是后续更高保真数据源。
    13-AU v2 建议输出到 facs_decoder_ext_v2.pt，与旧 5-AU / 11-AU 权重隔离。
    `seed` 固定初始化，保证可复现；落盘时另写 `<out>.json` provenance sidecar
    （轮数/lr/种子/数据/commit），`.pt` 仍是裸 state_dict、格式不变。
    """
    if extended:
        x, y = load_facs_csv_ext(csv_path)
    else:
        x, y = load_facs_csv(csv_path)
    torch.manual_seed(seed)
    model = FacsDecoder(hidden=hidden, num_layers=num_layers, extended=extended)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0.0
    epochs_ran = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        epochs_ran = epoch + 1
        if epoch % 50 == 0:
            logger.info("epoch %d loss %.6f (n=%d)", epoch, final_loss, x.shape[0])

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved facs decoder -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_facs.py",
        model=model,
        model_config={"hidden": hidden, "num_layers": num_layers, "extended": extended},
        data_config={},
        data_source=csv_path,
        n_samples=int(x.shape[0]),
        seed=seed,
        lr=lr,
        epochs_requested=epochs,
        epochs_ran=epochs_ran,
        final_train_loss=final_loss,
    )
    return final_loss


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="FACS 标注 CSV 路径")
    parser.add_argument("--epochs", type=int, default=300)
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
        epochs=args.epochs,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=out_path,
        extended=args.ext,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
