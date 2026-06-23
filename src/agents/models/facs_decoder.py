"""FacsDecoder：(v,a) → FACS 动作单元强度（从 AffectNet/DISFA 等学）。

替换 ExpressionDecoder 中"facs_au"通道的合成占位，用真实 AU 标注训练。
AU 顺序与占位一致：[AU04, AU06, AU12, AU15, intensity]，均归一化 [0,1]。
"""

from __future__ import annotations

import torch
from torch import nn

FACS_DIM = 5  # [AU04, AU06, AU12, AU15, intensity]
FACS_KEYS = ["AU04", "AU06", "AU12", "AU15", "intensity"]


class FacsDecoder(nn.Module):
    """(v,a) → 5 维 AU 强度的 MLP，sigmoid 输出 [0,1]。"""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, FACS_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_facs(self, valence: float, arousal: float) -> dict[str, float]:
        """单点推理，返回 facs_au 通道字典。"""
        self.eval()
        with torch.no_grad():
            vec = self(torch.tensor([[valence, arousal]], dtype=torch.float32))[0].tolist()
        return dict(zip(FACS_KEYS, vec, strict=True))


def load_facs_decoder(path: str, hidden: int = 16) -> FacsDecoder:
    """从权重文件加载已训练的 FACS 解码器。"""
    model = FacsDecoder(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
