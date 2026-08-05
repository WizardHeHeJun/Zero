"""MotionDecoder：(v,a) → 运动学调制系数（幅度/速度/onset）。

与既有输出侧解码器同族，但**结构上刻意不同**——议会数学席判定：RAVDESS 的 (v,a) 标签
只落在有限个离散锚点上（强度修正后 15 个），定义域支撑集基数固定 ⇒ **无法唯一确定一个
二元连续函数**，锚点之间的形状纯由归纳偏置决定；而 `Linear(2→16→K)` 的参数量 48+17K
对任意 K 都远超锚点数，属结构性过参数化。

故本模块提供两种，默认用插值版：

- `AnchorInterpolator`（默认）：自由度 = 锚点数，用高斯 RBF 在锚点间插值。参数量与数据
  规模匹配，不冒充"学到了连续函数"；且天然满足"锚点上贴合、锚点间平滑"。
- `MotionMLP`（对比用）：沿用既有族的小 MLP，供消融对照——想验证"插值是否真的更稳"时用。

⚠ 输出**不过 Sigmoid**：既有 4 个解码器末层统一 `Sigmoid→[0,1]` 是因为它们的目标
（AU 强度/生理/韵律）本就归一化到 [0,1]；调制系数是**乘性缩放**，压进 [0,1] 等于禁止
放大，与语义不符。改用 `softplus` 保正性（幅度/速度/锐度均非负），值域 (0,∞)。
"""

from __future__ import annotations

import torch
from torch import nn


class AnchorInterpolator(nn.Module):
    """高斯 RBF 锚点插值：自由度 = 锚点数，规避 8/15 锚点下的结构性不可辨识。

    每个锚点持有一组可学的调制系数；预测 = 按到各锚点的距离做 softmax 加权平均。
    `log_bandwidth` 可学，让模型自己决定插值的平滑程度（带宽大→接近全局均值，
    带宽小→接近最近邻）。

    参数量 = 锚点数 × K + 1，**恒不超过**观测到的锚点数 × K —— 这正是数学席要求的
    「参数量与数据规模匹配」。
    """

    def __init__(self, anchors: torch.Tensor, out_dim: int) -> None:
        """
        Args:
            anchors: 形状 (A, 2) 的锚点坐标（训练集里出现过的 (v,a) 去重）。
            out_dim: 调制系数维数 K。
        """
        super().__init__()
        if anchors.ndim != 2 or anchors.shape[1] != 2:
            raise ValueError(f"anchors 应为 (A, 2)，实得 {tuple(anchors.shape)}")
        # 先声明类型再 register_buffer：否则 `nn.Module.__getattr__` 的返回类型让
        # mypy 解析不出 Tensor（buffer 属性的既知类型检查坑）。
        self.anchors: torch.Tensor
        self.register_buffer("anchors", anchors.clone())
        self.values = nn.Parameter(torch.zeros(anchors.shape[0], out_dim))
        # 初始带宽 exp(0)=1.0：略大于 (v,a) 平面上锚点的典型间距，起步偏平滑
        # （宁可一开始接近全局均值，也不要一开始就退化成最近邻）；随训练自适应。
        self.log_bandwidth = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 2) → (N, K)。"""
        # (N, A)：到每个锚点的平方距离
        diff = x.unsqueeze(1) - self.anchors.unsqueeze(0)
        sq_dist = (diff * diff).sum(dim=-1)
        bandwidth = torch.exp(self.log_bandwidth).clamp(min=1e-3)
        weights = torch.softmax(-sq_dist / (2.0 * bandwidth * bandwidth), dim=-1)
        return nn.functional.softplus(weights @ self.values)


class MotionMLP(nn.Module):
    """小 MLP 对照组（沿用既有解码器族的形状），供「插值 vs MLP」消融。

    ⚠ 在 15 锚点上它是过参数化的（见模块 docstring），默认不用；留着是为了让
    「插值更稳」这个判断有可核验的对照，而不是一句断言。
    """

    def __init__(self, out_dim: int, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out
