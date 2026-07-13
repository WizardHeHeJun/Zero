"""FacsDecoder：(v,a) → FACS 动作单元强度（从 AffectNet/DISFA 等学）。

替换 ExpressionDecoder 中"facs_au"通道的合成占位，用真实 AU 标注训练。
AU 顺序与占位一致：[AU04, AU06, AU12, AU15, intensity]，均归一化 [0,1]。

扩展集（env 门控，默认关·零回归）：FACS_KEYS_EXT / FACS_DIM_EXT=11，
加 AU01/02/05/07/20/23，覆盖愤怒(coping>0)与恐惧(coping<0)的区分性 AU。
真权重文件：facs_decoder_ext.pt（当前外部数据阻塞，待 AffectNet/DISFA AU 标注，Q3）。
"""

from __future__ import annotations

import torch
from torch import nn

FACS_DIM = 5  # [AU04, AU06, AU12, AU15, intensity]  ← 不改，零回归
FACS_KEYS = ["AU04", "AU06", "AU12", "AU15", "intensity"]  # ← 不改，零回归

# 扩展集（议会设计门 PASS·路径 b）：独立常量，不覆盖旧集合
# AU01/02：额肌，恐惧扬眉（coping<0 驱动，需联动同升）
# AU04：皱眉，愤怒/恐惧共有（保留旧向）
# AU05：上睑抬，高唤醒共有
# AU06/AU12：正效价激活（保留旧向）
# AU07：睑紧，高唤醒×负效价
# AU15：负效价压嘴角（保留旧向）
# AU20：笑肌横拉唇，恐惧（coping<0 驱动）
# AU23：口轮匝肌唇紧，愤怒对抗准备（coping>0 驱动；跨文化普遍性有争议）
# intensity：全局强度（保留旧向）
FACS_DIM_EXT = 11
FACS_KEYS_EXT = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU12",
    "AU15",
    "AU20",
    "AU23",
    "intensity",
]


class FacsDecoder(nn.Module):
    """(v,a) → AU 强度的 MLP，sigmoid 输出 [0,1]。

    `extended=False`（默认）：5 维旧集合（FACS_KEYS），零回归。
    `extended=True`：11 维扩展集合（FACS_KEYS_EXT），真权重待数据（Q3）。
    """

    def __init__(
        self,
        hidden: int = 16,
        num_layers: int = 1,
        extended: bool = False,
    ) -> None:
        super().__init__()
        self.extended = extended
        out_dim = FACS_DIM_EXT if extended else FACS_DIM
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_facs(self, valence: float, arousal: float) -> dict[str, float]:
        """单点推理，返回 facs_au 通道字典（键集由 extended 决定）。"""
        keys = FACS_KEYS_EXT if self.extended else FACS_KEYS
        self.eval()
        with torch.no_grad():
            vec = self(torch.tensor([[valence, arousal]], dtype=torch.float32))[0].tolist()
        return dict(zip(keys, vec, strict=True))


def load_facs_decoder(
    path: str,
    hidden: int = 16,
    num_layers: int = 1,
    extended: bool = False,
) -> FacsDecoder:
    """从权重文件加载已训练的 FACS 解码器。

    extended=True 时对应 facs_decoder_ext.pt（11 维；真权重待数据，Q3）。
    """
    model = FacsDecoder(hidden=hidden, num_layers=num_layers, extended=extended)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
