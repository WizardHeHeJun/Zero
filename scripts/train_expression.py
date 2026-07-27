"""训练 ExpressionDecoder：把 (v,a)→通道 解析占位蒸馏成可训练网络（合成数据 bootstrap）。

用法：python -m scripts.train_expression --epochs 300 --out artifacts/expression_decoder.pt
真数据到位后，把 synthetic_pairs 换成真实 DataLoader 即可，本训练循环复用。

注意：训更大网络（--hidden/--num-layers 非默认值）需从头重训；
原 Release 权重仅兼容默认形状（hidden=32, num_layers=2）。

──────────────────────────────────────────────────────────────────
canonical_physiology 重训说明（议会 2026-07-23）：
  ExpressionDecoder 的蒸馏目标由 affect_to_vector() 生成（11 维向量）。
  idx7 有双语义：
    门关（默认·legacy）：pupil_n = clamp(arousal, 0, 1)
    门开（canonical）  ：temperature_n = (36−3·clamp(|arousal|)−30)/10

  门关（默认）= legacy 目标 → 旧权重（expression_decoder.pt）兼容，零回归；
  门开（canonical）= canonical 目标 → 用 --canonical-physiology 旗标重训（默认输出
  artifacts/expression_decoder_canonical.pt）：
    python -m scripts.train_expression --epochs 300 --canonical-physiology

  该旗标经 train() 透传 synthetic_pairs → affect_to_vector(canonical_physiology=True)，
  蒸馏目标即 canonical 布局（idx7=temperature_n）。旧权重（expression_decoder.pt）不得在
  canonical 路径复用：idx7 会被误解为 temperature，导致生理通道量纲错误。canonical demo
  权重按需产出（生产 MCP/chat 走 decode_channels 不依赖蒸馏 MLP）。
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from scripts._train_common import write_provenance
from src.agents.datasets.synthetic import synthetic_pairs
from src.agents.models.expression_decoder import ExpressionDecoder

logger = logging.getLogger(__name__)


def train(
    *,
    epochs: int = 300,
    n: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 32,
    num_layers: int = 2,
    out: str = "artifacts/expression_decoder.pt",
    canonical_physiology: bool = False,
) -> float:
    """全批量训练解码器并保存权重，返回最终 MSE 损失。

    `canonical_physiology`（默认 False=legacy 目标·零回归）透传 `synthetic_pairs` →
    `affect_to_vector`：True 时蒸馏 canonical 布局（idx7=temperature_n），须配 canonical 输出路径。
    `seed` 同时用于合成数据生成（`synthetic_pairs`）与模型初始化，保证可复现；落盘时另写
    `<out>.json` provenance sidecar（轮数/lr/种子/蒸馏口径/commit），`.pt` 格式不变。
    """
    x, y = synthetic_pairs(n, seed=seed, canonical_physiology=canonical_physiology)
    torch.manual_seed(seed)
    model = ExpressionDecoder(hidden=hidden, num_layers=num_layers)
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
            logger.info("epoch %d loss %.6f", epoch, final_loss)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("saved decoder -> %s (final loss %.6f)", out_path, final_loss)
    write_provenance(
        out_path,
        script="scripts/train_expression.py",
        model=model,
        model_config={"hidden": hidden, "num_layers": num_layers},
        # canonical_physiology 决定 idx7 是 pupil_n 还是 temperature_n——两种口径的权重
        # 不可互换（见本模块 docstring），必须随权重一起落账。
        data_config={"n": n, "canonical_physiology": canonical_physiology},
        data_source=None,
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--canonical-physiology",
        action="store_true",
        help="蒸馏 canonical physiology 布局（idx7=temperature_n）；默认输出 _canonical.pt",
    )
    parser.add_argument("--seed", type=int, default=0, help="固定初始化，保证可复现")
    args = parser.parse_args()
    # 未显式给 --out 时按目标口径选默认路径，避免 canonical 权重覆写 legacy 权重（口径不兼容）。
    out = args.out or (
        "artifacts/expression_decoder_canonical.pt"
        if args.canonical_physiology
        else "artifacts/expression_decoder.pt"
    )
    final = train(
        epochs=args.epochs,
        n=args.n,
        hidden=args.hidden,
        num_layers=args.num_layers,
        out=out,
        canonical_physiology=args.canonical_physiology,
        seed=args.seed,
    )
    print(f"done, final loss={final:.6f}")


if __name__ == "__main__":
    main()
