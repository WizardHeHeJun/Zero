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
    """句向量 → affect 维度的 MLP 头，tanh 输出落在 [-1, 1]。

    默认模式（finetune_encoder=False）：编码器不是本 module 成员（经 encode_texts 单例
    延迟加载，冻结作特征提取器），state_dict 仅含 MLP 头；CPU 可行。

    finetune_encoder=True 时：SentenceTransformer 作 nn.Module 成员持有，
    parameters() 含编码器，可端到端 backward；需 GPU，MiniLM ~22M CPU 极慢。

    output_dim 控制输出维度：2=VA（默认，零回归）/ 1=D（单维）/ 3=VAD（全维）。
    默认 finetune_encoder=False + output_dim=2 时 state_dict 键集与批 2 后版本逐字相同。
    """

    def __init__(
        self,
        dim: int = ST_FEATURE_DIM,
        hidden: int = 64,
        num_layers: int = 1,
        *,
        encoder: str = DEFAULT_ENCODER,
        finetune_encoder: bool = False,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.encoder_name = encoder
        self.finetune_encoder = finetune_encoder
        self.output_dim = output_dim

        if finetune_encoder:
            from sentence_transformers import SentenceTransformer as _ST

            self.encoder_module: nn.Module | None = _ST(encoder)
        else:
            self.encoder_module = None

        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, output_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_texts(self, texts: list[str]) -> torch.Tensor:
        """finetune_encoder=True 时的端到端前向：文本→编码器(带梯度)→MLP 头。

        默认冻结路径用 forward(预计算句向量)；本方法把编码器纳入计算图，使其参数参与
        backward——真正端到端微调（需 GPU；CPU 仅供极小 smoke）。`forward` 收预计算张量时
        编码器不在图内、encoder 拿不到梯度（W4：故微调路径必须走此方法，非 forward）。
        """
        if self.encoder_module is None:
            raise RuntimeError(
                "forward_texts 仅在 finetune_encoder=True 时可用（需持 encoder_module）"
            )
        # tokenize/forward 是 SentenceTransformer 的动态 API，不在 nn.Module 静态类型上。
        features = self.encoder_module.tokenize(texts)  # type: ignore[operator]
        device = next(self.parameters()).device
        # tokenize 输出的 features 里非张量值（如 metadata）无 .to，跨 ST 版本安全搬运张量。
        features = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()}
        emb = self.encoder_module(features)["sentence_embedding"]
        return self.net(emb)

    def predict_affect(self, text: str) -> tuple[float, float]:
        """单条文本推理，返回 (valence, arousal)。"""
        self.eval()
        with torch.no_grad():
            feats = encode_texts([text], encoder=self.encoder_name)
            out = self(feats)[0].tolist()
        return (out[0], out[1])


def load_st_text_affect_regressor(
    path: str,
    dim: int = ST_FEATURE_DIM,
    hidden: int = 64,
    num_layers: int = 1,
    *,
    encoder: str = DEFAULT_ENCODER,
    output_dim: int = 2,
) -> STTextAffectRegressor:
    """从权重文件加载已训练的句向量文本情感回归器（仅 MLP 头）。"""
    model = STTextAffectRegressor(
        dim=dim,
        hidden=hidden,
        num_layers=num_layers,
        encoder=encoder,
        output_dim=output_dim,
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
