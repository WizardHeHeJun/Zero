"""合成数据：(v,a) 均匀采样 + 解析"真值"通道向量，用于 bootstrap 训练管线。

真数据到位后，新增同形 (X, Y) 的真实 DataLoader 替换本模块即可，模型/训练复用。
"""

from __future__ import annotations

import torch

from src.agents.models.expression_decoder import affect_to_vector


def synthetic_pairs(
    n: int, *, seed: int = 0, canonical_physiology: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 n 对 (X=(v,a), Y=11 维归一化通道向量)。

    X ∈ [-1,1]^2，Y ∈ [0,1]^11。返回 (X, Y) 两个 float32 张量。

    `canonical_physiology` 透传 `affect_to_vector` 的 physiology 口径（默认 False=legacy·零回归·
    与旧 Release 权重兼容；True=canonical 布局 idx7=temperature_n，供 canonical demo 权重蒸馏，
    见 `scripts/train_expression.py --canonical-physiology`）。
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=generator) * 2.0 - 1.0
    y = torch.tensor(
        [
            affect_to_vector(float(v), float(a), canonical_physiology=canonical_physiology)
            for v, a in x
        ],
        dtype=torch.float32,
    )
    return x, y
