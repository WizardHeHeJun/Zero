"""ExpressionDecoder：把 (valence, arousal) → 表达通道 的解析占位蒸馏成可训练 torch 网络。

向量布局（11 维，全部归一化到 [0,1]，与 affect_math 的解析映射一致）：
  0 AU04 · 1 AU06 · 2 AU12 · 3 AU15 · 4 au_intensity
  5 hr_n · 6 gsr · 7 pupil_n
  8 speech_rate_n · 9 pitch_n · 10 energy

`affect_to_vector` 是蒸馏目标（解析"真值"）；`vector_to_channels` 把网络输出反归一化回
与 `affect_math.decode_channels` 同构的通道字典（不含 text_label，由调用方补）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.agents.affect_math import clamp, text_label

CHANNEL_DIM = 11


def affect_to_vector(valence: float, arousal: float) -> list[float]:
    """解析"真值"：(v,a) → 11 维归一化通道向量（蒸馏目标）。"""
    return [
        clamp(-0.6 * valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU04
        clamp(0.6 * valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU06
        clamp(valence, 0.0, 1.0) if valence >= 0 else 0.0,  # AU12
        clamp(-valence, 0.0, 1.0) if valence < 0 else 0.0,  # AU15
        clamp(abs(arousal), 0.0, 1.0),  # au_intensity
        clamp(arousal, 0.0, 1.0),  # hr_n
        clamp(
            abs(arousal), 0.0, 1.0
        ),  # gsr（议会 B-2/A-P0-D：EDA 随 |arousal|，与 decode_channels 一致）
        clamp(arousal, 0.0, 1.0),  # pupil_n
        (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # speech_rate_n
        (clamp(arousal, -1.0, 1.0) + 1.0) / 2.0,  # pitch_n（议会 B-8/A-P0-C：F0 随唤醒非效价）
        clamp(0.5 + 0.5 * arousal, 0.0, 1.0),  # energy
    ]


def vector_to_channels(vec: list[float]) -> dict[str, Any]:
    """把 11 维归一化向量反归一化为通道字典（结构对齐 decode_channels，不含 text_label）。"""
    return {
        "facs_au": {
            "AU04": vec[0],
            "AU06": vec[1],
            "AU12": vec[2],
            "AU15": vec[3],
            "intensity": vec[4],
        },
        "physiology": {
            "heart_rate_bpm": 70.0 + 40.0 * vec[5],
            "skin_conductance": vec[6],
            "pupil_mm": 3.0 + 2.0 * vec[7],
        },
        "prosody": {
            "speech_rate": 1.0 + 0.5 * (2.0 * vec[8] - 1.0),
            "pitch": 1.0 + 0.3 * (2.0 * vec[9] - 1.0),
            "energy": vec[10],
        },
    }


class ExpressionDecoder(nn.Module):
    """(v,a) → 11 维通道向量的 MLP，输出经 sigmoid 落在 [0,1]。"""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, CHANNEL_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        """单点推理，返回与 decode_channels 同构的通道字典（含 text_label）。"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[valence, arousal]], dtype=torch.float32)
            vec = self(x)[0].tolist()
        channels = vector_to_channels(vec)
        channels["text_label"] = text_label(valence, arousal)
        return channels


def load_decoder(path: str, hidden: int = 32) -> ExpressionDecoder:
    """从权重文件加载已训练的解码器。"""
    model = ExpressionDecoder(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
