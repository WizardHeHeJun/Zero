"""ProsodyDecoder：(v,a) → 真实韵律特征（从 RAVDESS 学）。

替换 ExpressionDecoder 中"韵律"通道的合成占位，用真实语音特征训练。
通道复合见 `composite.py` 的 CompositeChannelDecoder。
"""

from __future__ import annotations

import torch
from torch import nn

PROSODY_DIM = 3  # [speech_rate, pitch, energy]，均归一化到 [0,1]


class ProsodyDecoder(nn.Module):
    """(v,a) → 3 维归一化韵律特征的 MLP，sigmoid 输出 [0,1]。"""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, PROSODY_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_prosody(self, valence: float, arousal: float) -> dict[str, float]:
        """单点推理，返回韵律通道字典（值为归一化 [0,1] 的真实学习特征）。"""
        self.eval()
        with torch.no_grad():
            vec = self(torch.tensor([[valence, arousal]], dtype=torch.float32))[0].tolist()
        return {"speech_rate": vec[0], "pitch": vec[1], "energy": vec[2]}


def load_prosody_decoder(path: str, hidden: int = 16) -> ProsodyDecoder:
    """从权重文件加载已训练的韵律解码器。"""
    model = ProsodyDecoder(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
