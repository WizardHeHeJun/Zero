"""EmoBank 文本→V-A DataLoader（输入侧）。

获取数据（开放）：
  https://github.com/JULIELab/EmoBank （emobank.csv）或 Kaggle 镜像；放 data/emobank.csv。
  列：id,split,V,A,D,text （V/A/D 为 1~5 量表，3 为中性）。
  归一化：valence=(V-3)/2、arousal=(A-3)/2 → [-1,1]。
训练：python -m scripts.train_text_affect --csv data/emobank.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.agents.affect_math import clamp
from src.agents.models.text_affect_regressor import TEXT_FEATURE_DIM, hash_features


def read_emobank_rows(
    path: str | Path, *, limit: int | None = None
) -> tuple[list[str], list[list[float]]]:
    """读取 EmoBank CSV，返回 (texts, Y=(v,a))。V/A 由 1~5 归一化到 [-1,1]。

    供哈希词袋与句向量两种 loader 共用同一数据解析/归一化源。无数据行时抛 ValueError。
    """
    texts: list[str] = []
    ys: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            ys.append(
                [
                    clamp((float(row["V"]) - 3.0) / 2.0, -1.0, 1.0),
                    clamp((float(row["A"]) - 3.0) / 2.0, -1.0, 1.0),
                ]
            )
            if limit is not None and len(texts) >= limit:
                break

    if not texts:
        raise ValueError(f"{path} 无可用数据行（需含 V,A,text 列）")
    return texts, ys


def load_emobank(
    path: str | Path, *, dim: int = TEXT_FEATURE_DIM, limit: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 EmoBank CSV，返回 (X=哈希词袋特征, Y=(v,a)) float32 张量。

    X ∈ [0,1]^dim，Y ∈ [-1,1]^2。无数据行时抛 ValueError。
    """
    texts, ys = read_emobank_rows(path, limit=limit)
    xs = [hash_features(text, dim) for text in texts]
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
