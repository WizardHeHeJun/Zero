"""PhysiologyDecoder：(v,a) → 真实生理特征（从 WESAD 学）。

替换 ExpressionDecoder 中"生理"通道的合成占位，用真实穿戴信号特征训练。
输出 [hr, eda, temp] 归一化，predict_physiology 反归一化为可读单位。
"""

from __future__ import annotations

import torch
from torch import nn

PHYSIOLOGY_DIM = 3  # [hr, eda, temp]，均归一化到 [0,1]


class PhysiologyDecoder(nn.Module):
    """(v,a) → 3 维归一化生理特征的 MLP，sigmoid 输出 [0,1]。"""

    def __init__(self, hidden: int = 16, num_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, PHYSIOLOGY_DIM), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_physiology(self, valence: float, arousal: float) -> dict[str, float]:
        """单点推理，返回生理通道字典（反归一化为可读单位）。"""
        self.eval()
        with torch.no_grad():
            vec = self(torch.tensor([[valence, arousal]], dtype=torch.float32))[0].tolist()
        return {
            "heart_rate_bpm": 50.0 + vec[0] * 70.0,
            "skin_conductance": vec[1] * 20.0,
            "temperature_c": 30.0 + vec[2] * 10.0,
        }


def load_physiology_decoder(path: str, hidden: int = 16, num_layers: int = 1) -> PhysiologyDecoder:
    """从权重文件加载已训练的生理解码器。"""
    model = PhysiologyDecoder(hidden=hidden, num_layers=num_layers)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
