"""在 FACS 标注 CSV 上训练 FacsDecoder（(v,a)→AU 强度），保存权重。

用法（旧 5-AU）：python -m scripts.train_facs --csv data/facs/labels.csv --epochs 300
用法（11-AU 扩展）：python -m scripts.train_facs --csv data/facs/labels_ext.csv --ext

数据获取/CSV 格式见 src/agents/datasets/facs_csv.py 模块文档。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=16, num_layers=1）。

--ext 模式说明：
  输出文件名 facs_decoder_ext.pt（隔离命名，不覆盖旧 facs_decoder.pt）。
  数据阻塞：需 AffectNet/DISFA 含 AU01/02/05/07/20/23 完整标注（外部 EULA，Q3 等待）。
  真权重尚未可训；此 --ext 模式仅为脚手架占位，数据就绪后解除阻塞即可运行。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

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
) -> float:
    """加载 FACS CSV、全批量训练 FacsDecoder 并保存权重，返回最终 MSE。

    extended=True：加载 11-AU 扩展 CSV（load_facs_csv_ext），用 FacsDecoder(extended=True)，
    输出文件名应指向 facs_decoder_ext.pt（隔离命名，CS 席约束 #7）。
    数据阻塞：extended=True 时需外部 AU 标注（Q3 等待 EULA）。
    """
    if extended:
        x, y = load_facs_csv_ext(csv_path)
    else:
        x, y = load_facs_csv(csv_path)
    model = FacsDecoder(hidden=hidden, num_layers=num_layers, extended=extended)
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
    logger.info("saved facs decoder -> %s (final loss %.6f)", out_path, final_loss)
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
        help="训练 11-AU 扩展模型（数据阻塞，Q3；输出 facs_decoder_ext.pt）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "权重输出路径"
            "（默认：--ext→artifacts/facs_decoder_ext.pt；否则→artifacts/facs_decoder.pt）"
        ),
    )
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
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
