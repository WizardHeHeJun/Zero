"""STTextAffectRegressor：文本 → (valence, arousal)，前端用预训练句向量编码器。

相比 text_affect_regressor.py 的哈希词袋（无语义泛化、跨域即失效），句向量带语义，
对未见词/换体裁更稳（见 ECIR'23 多语言 transformer VAD 回归）。编码器**冻结**作特征
提取器，只训上面的 MLP 头——CPU 可行、不易过拟合，干净隔离出"语义表示 vs 词袋"这一变量。

依赖 sentence-transformers（ml 重依赖，延迟 import；编排层/默认路径不引入）。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "all-MiniLM-L6-v2"
ST_FEATURE_DIM = 384  # all-MiniLM-L6-v2 句向量维


@lru_cache(maxsize=4)
def _load_encoder(name: str = DEFAULT_ENCODER) -> SentenceTransformer:
    """加载并缓存句向量编码器（进程内单例，避免重复加载权重）。"""
    from sentence_transformers import SentenceTransformer

    logger.info("loading sentence encoder %s", name)
    return SentenceTransformer(name)


def encode_texts(texts: list[str], *, encoder: str = DEFAULT_ENCODER) -> torch.Tensor:
    """把文本批量编码成句向量张量 (n, dim)，float32，L2 归一化。"""
    model = _load_encoder(encoder)
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return torch.tensor(vecs, dtype=torch.float32)


class STTextAffectRegressor(nn.Module):
    """冻结句向量 → (valence, arousal) 的 MLP 头，tanh 输出落在 [-1, 1]。

    编码器不是本 module 成员（经 encode_texts 单例延迟加载），故 state_dict 仅含 MLP 头。
    """

    def __init__(
        self, dim: int = ST_FEATURE_DIM, hidden: int = 64, *, encoder: str = DEFAULT_ENCODER
    ) -> None:
        super().__init__()
        self.dim = dim
        self.encoder_name = encoder
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_affect(self, text: str) -> tuple[float, float]:
        """单条文本推理，返回 (valence, arousal)。"""
        self.eval()
        with torch.no_grad():
            feats = encode_texts([text], encoder=self.encoder_name)
            out = self(feats)[0].tolist()
        return (out[0], out[1])


def load_st_text_affect_regressor(
    path: str, dim: int = ST_FEATURE_DIM, hidden: int = 64, *, encoder: str = DEFAULT_ENCODER
) -> STTextAffectRegressor:
    """从权重文件加载已训练的句向量文本情感回归器（仅 MLP 头）。"""
    model = STTextAffectRegressor(dim=dim, hidden=hidden, encoder=encoder)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
