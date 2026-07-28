"""FACS 表情 DataLoader（CSV）：(v,a) + AU 强度。

获取数据（需 EULA）：AffectNet / DISFA / EmotioNet 申请后，把标注导出为 CSV，列：
  valence,arousal,AU04,AU06,AU12,AU15,intensity
  （valence/arousal ∈ [-1,1]；AU ∈ [0,1]）
说明：AffectNet 提供 valence/arousal；DISFA/EmotioNet 提供 AU 强度——按你的数据 join 后
导出本 CSV。放到例如 data/facs/labels.csv（已 gitignore）。
训练：python -m scripts.train_facs --csv data/facs/labels.csv

扩展集（13-AU）独立 loader：
  CSV 列：valence,arousal + FACS_KEYS_EXT 全部键（当前 13 键：AU01,AU02,AU04,AU05,AU06,
  AU07,AU12,AU15,AU17,AU20,AU23,AU26,intensity；实现按 FACS_KEYS_EXT 动态取，随其演进）。
  EULA-free 路径（已可训，无需 AffectNet/DISFA）：laion/emonet-face-binary（CC-BY-4.0）
  → scripts/build_emonet_dataset → OpenFace 抽 AU → scripts/build_facs_ext_csv。
  训练：python -m scripts.train_facs --csv data/facs/labels_ext.csv --ext
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.agents.affect_math import clamp
from src.agents.models.facs_decoder import FACS_KEYS, FACS_KEYS_EXT


def load_facs_csv(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 FACS 标注 CSV，返回 (X=(v,a), Y=AU 5维) float32 张量。

    X ∈ [-1,1]^2，Y ∈ [0,1]^5。无数据行时抛 ValueError。
    旧签名不变（零回归；CS 席约束 #9）。
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


def load_facs_csv_ext(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """读取扩展集 FACS 标注 CSV，返回 (X=(v,a), Y=AU 强度) float32 张量。

    X ∈ [-1,1]^2，Y ∈ [0,1]^len(FACS_KEYS_EXT)（当前 13 维）。无数据行时抛 ValueError。

    CSV 需含 valence,arousal + FACS_KEYS_EXT 全部键（按该常量动态取，随其演进不必改此处）。
    EULA-free 数据路径见模块 docstring（emonet CC-BY + OpenFace，已产出真权重）。
    此 loader 独立于 load_facs_csv，不改旧签名（CS 席约束 #9）。
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
            ys.append([clamp(float(row[key]), 0.0, 1.0) for key in FACS_KEYS_EXT])

    if not xs:
        raise ValueError(
            f"{path} 无可用数据行（需含 valence,arousal,{','.join(FACS_KEYS_EXT)} 列）"
        )
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
