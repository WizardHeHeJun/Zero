"""DirectionHead：句向量 → 单 logit，用于 motivational_direction_prior 推理。

训练侧见 scripts/train_direction_head.py（BCEWithLogitsLoss on raw logit；末层无 Tanh）。
推理侧：tanh(logit) ∈ [-1, 1]，+1≈anger(趋近)、-1≈fear(回避)。

延迟 import torch（模块顶层不引重依赖；仅在加载权重时才 import）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# 与 text_affect_regressor_st.py 对齐：all-MiniLM-L6-v2 句向量维
_ST_FEATURE_DIM = 384


class DirectionHead:
    """句向量 → 单 logit（无末层 Tanh；推理时 tanh(logit) ∈ [-1, 1]）。

    内部持有 nn.Module 实例；顶层类不继承 nn.Module 以避免在无 torch 环境下
    导入即报错——nn.Module 依赖延迟在 load_direction_head 工厂中引入。

    Attributes:
        net: 内部 nn.Module 实例（_DirectionHeadModule），类型为 Any（延迟 import）。
    """

    def __init__(self, net: Any) -> None:
        self.net: Any = net

    def __call__(self, x: Any) -> Any:
        """前向推理，返回 raw logit（shape=(batch,)）。"""
        return self.net(x)


def _build_module(dim: int, hidden: int = 64) -> Any:
    """构造内部 nn.Module（延迟 import torch/nn）。返回 Any 以对齐延迟类型。"""
    import torch.nn as nn

    class _DirectionHeadModule(nn.Module):
        """句向量 → 单 logit（无末层 Tanh；BCEWithLogitsLoss 训练；推理 tanh(logit)∈[-1,1]）。"""

        def __init__(self, dim: int, hidden: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)  # type: ignore[no-any-return]  # raw logit，shape=(batch,)

    return _DirectionHeadModule(dim=dim, hidden=hidden)


def load_direction_head(path: str, dim: int = _ST_FEATURE_DIM) -> DirectionHead:
    """从权重文件加载已训练的 DirectionHead（仅 MLP 头）。

    仿 load_st_text_affect_regressor：构造 → load_state_dict(map_location=cpu) → eval。

    Args:
        path: 权重文件路径（.pt）。
        dim: 句向量维度，默认 384（all-MiniLM-L6-v2）。

    Returns:
        处于 eval 模式的 DirectionHead 实例。

    Raises:
        任何 torch/IO 异常直接向上抛出，由调用方 fail-soft 处理。
    """
    import torch

    module: Any = _build_module(dim=dim)
    state = torch.load(path, map_location="cpu")
    module.load_state_dict(state)
    module.eval()
    logger.info("已加载 DirectionHead，权重路径=%s dim=%d", path, dim)
    return DirectionHead(net=module)
