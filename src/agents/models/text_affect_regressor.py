"""TextAffectRegressor：文本 → (valence, arousal)（输入侧，从 EmoBank 学）。

这是输入侧的真网络化：把自然语言映射到情感维度，供未来"文本输入"的 PerceptionAgent 使用。
不依赖预训练编码器——用稳定哈希词袋（hashlib）做轻量特征，纯 torch 训练。
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn

TEXT_FEATURE_DIM = 256


def hash_features(text: str, dim: int = TEXT_FEATURE_DIM) -> list[float]:
    """稳定哈希词袋特征（跨进程确定，用 md5），按词数归一化。"""
    vec = [0.0] * dim
    for token in text.lower().split():
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    total = sum(vec) or 1.0
    return [v / total for v in vec]


class TextAffectRegressor(nn.Module):
    """哈希词袋 → (valence, arousal) 的 MLP，tanh 输出落在 [-1, 1]。"""

    def __init__(self, dim: int = TEXT_FEATURE_DIM, hidden: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.dim = dim
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 2), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_affect(self, text: str) -> tuple[float, float]:
        """单条文本推理，返回 (valence, arousal)。"""
        self.eval()
        with torch.no_grad():
            feats = torch.tensor([hash_features(text, self.dim)], dtype=torch.float32)
            out = self(feats)[0].tolist()
        return (out[0], out[1])


def load_text_affect_regressor(
    path: str, dim: int = TEXT_FEATURE_DIM, hidden: int = 64, num_layers: int = 1
) -> TextAffectRegressor:
    """从权重文件加载已训练的文本情感回归器。"""
    model = TextAffectRegressor(dim=dim, hidden=hidden, num_layers=num_layers)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
