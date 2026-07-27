"""EmoBank 文本→V-A DataLoader（句向量版，输入侧）。

复用 emobank.read_emobank_rows 的 CSV 解析/归一化，把哈希词袋换成预训练句向量
（带语义泛化、跨域更稳）。句向量编码是最慢的一步，训练前一次性预计算。
依赖 sentence-transformers（ml 重依赖，经 encode_texts 延迟 import）。
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.agents.datasets.emobank import read_emobank_rows
from src.agents.models.text_affect_regressor_st import DEFAULT_ENCODER, encode_texts


def load_emobank_embeddings(
    path: str | Path,
    *,
    encoder: str = DEFAULT_ENCODER,
    limit: int | None = None,
    include_d: bool = False,
    split: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 EmoBank CSV 并把文本编码成句向量，返回 (X=句向量, Y) float32 张量。

    X ∈ R^dim（编码器维）。默认 include_d=False 时 Y ∈ [-1,1]^2（零回归）；
    include_d=True 时 Y ∈ [-1,1]^3（含 D 维，见 read_emobank_rows 的 Note）。
    无数据行时（经 read_emobank_rows）抛 ValueError。

    `split` 语义与泄漏警告见 `read_emobank_rows`（默认 None＝全量）。⚠ 句向量通道此前
    **一直没有这个参数**，两个调用方（`train_text_affect_st` / `train_text_affect_d`）
    因此都在读全量训练——包括官方 dev/test。已发布的 `text_affect_regressor_st.pt`
    即产自那条路径，其 loss 是训练集拟合度、不可当泛化指标读。
    """
    texts, ys = read_emobank_rows(path, limit=limit, include_d=include_d, split=split)
    x = encode_texts(texts, encoder=encoder)
    return x, torch.tensor(ys, dtype=torch.float32)
