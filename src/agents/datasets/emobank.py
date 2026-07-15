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
    path: str | Path, *, limit: int | None = None, include_d: bool = False
) -> tuple[list[str], list[list[float]]]:
    """读取 EmoBank CSV，返回 (texts, Y)。V/A/D 由 1~5 归一化到 [-1,1]。

    供哈希词袋与句向量两种 loader 共用同一数据解析/归一化源。无数据行时抛 ValueError。

    Args:
        path: EmoBank CSV 路径。
        limit: 最多读取行数，None 表示全量。
        include_d: 默认 False（零回归），Y shape=(n,2) 只含 V/A；
            True 时 Y shape=(n,3)，第三列为 D，归一化公式同 V/A：
            clamp((D-3)/2, -1, 1)。

    Note:
        D 列为 SAM Dominance 标注（Bradley & Lang 1994 SAM 量表，感受量非评价前件）。
        议会 2026-07-15 裁定「有条件可用」（结 2026-07-13 #2）：须选 writer perspective
        + 社会支配子集筛选 + 验证集方向校验(≥80%) + 训后独立性门控 r(V,D_pred)≤0.50；
        未执行修正=不可用。见 notes/2026-07-15-text-coping-potential-emobank-d-council.md。
    """
    texts: list[str] = []
    ys: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            entry: list[float] = [
                clamp((float(row["V"]) - 3.0) / 2.0, -1.0, 1.0),
                clamp((float(row["A"]) - 3.0) / 2.0, -1.0, 1.0),
            ]
            if include_d:
                entry.append(clamp((float(row["D"]) - 3.0) / 2.0, -1.0, 1.0))
            ys.append(entry)
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
