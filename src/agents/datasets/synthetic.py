"""合成数据：(v,a) 均匀采样 + 解析"真值"通道向量，用于 bootstrap 训练管线。

真数据到位后，新增同形 (X, Y) 的真实 DataLoader 替换本模块即可，模型/训练复用。
"""

from __future__ import annotations

import torch

from src.agents.models.expression_decoder import affect_to_vector


def synthetic_pairs(n: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 n 对 (X=(v,a), Y=11 维归一化通道向量)。

    X ∈ [-1,1]^2，Y ∈ [0,1]^11。返回 (X, Y) 两个 float32 张量。
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=generator) * 2.0 - 1.0
    y = torch.tensor(
        [affect_to_vector(float(v), float(a)) for v, a in x],
        dtype=torch.float32,
    )
    return x, y
