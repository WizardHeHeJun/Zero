"""ProsodyDecoder：(v,a) → 真实韵律特征（从 RAVDESS 学），及把它合进表达通道的复合器。

ProsodyDecoder 替换 ExpressionDecoder 中"韵律"通道的合成占位，用真实语音特征训练。
CompositeChannelDecoder 把若干"真通道模型"叠加到解析占位之上：有真模型的通道用真模型，
其余通道回退 affect_math 解析占位——实现逐通道、无破坏的渐进真网络化。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.agents.affect_math import decode_channels, text_label

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


class CompositeChannelDecoder:
    """复合通道解码器：在解析占位之上，用已注入的真模型覆盖对应通道。

    满足 ExpressionAgent 的 ChannelDecoder 协议（predict_channels）。
    """

    def __init__(self, prosody_model: ProsodyDecoder | None = None) -> None:
        self.prosody_model = prosody_model

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        channels = decode_channels((valence, arousal))  # 解析占位提供全部 4 通道
        if self.prosody_model is not None:
            channels["prosody"] = self.prosody_model.predict_prosody(valence, arousal)
        channels["text_label"] = text_label(valence, arousal)
        return channels


def load_prosody_decoder(path: str, hidden: int = 16) -> ProsodyDecoder:
    """从权重文件加载已训练的韵律解码器。"""
    model = ProsodyDecoder(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
