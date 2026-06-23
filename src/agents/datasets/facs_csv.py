"""FACS 表情 DataLoader（CSV）：(v,a) + AU 强度。

获取数据（需 EULA）：AffectNet / DISFA / EmotioNet 申请后，把标注导出为 CSV，列：
  valence,arousal,AU04,AU06,AU12,AU15,intensity
  （valence/arousal ∈ [-1,1]；AU ∈ [0,1]）
说明：AffectNet 提供 valence/arousal；DISFA/EmotioNet 提供 AU 强度——按你的数据 join 后
导出本 CSV。放到例如 data/facs/labels.csv（已 gitignore）。
训练：python -m scripts.train_facs --csv data/facs/labels.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.agents.affect_math import clamp
from src.agents.models.facs_decoder import FACS_KEYS


def load_facs_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 FACS 标注 CSV，返回 (X=(v,a), Y=AU 5维) float32 张量。

    X ∈ [-1,1]^2，Y ∈ [0,1]^5。无数据行时抛 ValueError。
    """
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            xs.append(
                [
                    clamp(float(row["valence"]), -1.0, 1.0),
                    clamp(float(row["arousal"]), -1.0, 1.0),
                ]
            )
            ys.append([clamp(float(row[key]), 0.0, 1.0) for key in FACS_KEYS])

    if not xs:
        raise ValueError(f"{path} 无可用数据行（需含 valence,arousal,{','.join(FACS_KEYS)} 列）")
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
