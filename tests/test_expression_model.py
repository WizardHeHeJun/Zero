"""真网络化（三-1 合成 bootstrap）：合成数据 / 解码器 / 注入回归 / 训练 smoke。

torch 缺失时整文件跳过，保证无 ml 依赖的环境核心套件仍可跑。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.agents.affect_math import decode_channels  # noqa: E402
from src.agents.datasets.synthetic import synthetic_pairs  # noqa: E402
from src.agents.expression import ExpressionAgent  # noqa: E402
from src.agents.models.expression_decoder import (  # noqa: E402
    CHANNEL_DIM,
    ExpressionDecoder,
    affect_to_vector,
    load_decoder,
    vector_to_channels,
)
from src.orchestration.state import AffectState  # noqa: E402

CHANNELS = {"facs_au", "text_label", "physiology", "prosody"}


def test_synthetic_pairs_shapes_and_ranges() -> None:
    x, y = synthetic_pairs(64, seed=1)
    assert x.shape == (64, 2)
    assert y.shape == (64, CHANNEL_DIM)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_decoder_forward_shape_and_range() -> None:
    model = ExpressionDecoder()
    with torch.no_grad():
        out = model(torch.zeros(5, 2))
    assert out.shape == (5, CHANNEL_DIM)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_affect_to_vector_matches_analytic_physiology_prosody() -> None:
    # 模型路径（向量映射）与解析占位在连续通道上应数值一致。
    v, a = 0.6, 0.5
    model_ch = vector_to_channels(affect_to_vector(v, a))
    analytic = decode_channels((v, a))
    for channel in ("physiology", "prosody"):
        for key, value in analytic[channel].items():
            assert model_ch[channel][key] == pytest.approx(value, abs=1e-6), (channel, key)


def test_vector_path_prosody_scale_ratio() -> None:
    # zero-link Q1（2026-07-14）：整向量通路把内部 [0,1] 反归一化回倍率口径 → 量纲标记 "ratio"
    # （与专用 ProsodyDecoder 的 normalized 相区分）。
    assert vector_to_channels(affect_to_vector(0.6, 0.5))["prosody_scale"] == "ratio"
    assert ExpressionDecoder().predict_channels(0.6, 0.5)["prosody_scale"] == "ratio"


def test_expression_agent_injected_decoder_preserves_contract() -> None:
    agent = ExpressionAgent(decoder=ExpressionDecoder())  # 未训练也应满足契约
    state = AffectState(affect_sample=(0.8, 0.7))
    expr = agent(state)["expression"]
    assert expr["valence_arousal"] is not None
    for head in ("spontaneous", "voluntary"):
        assert CHANNELS.issubset(expr[head]), head


def test_train_smoke_reduces_loss_and_load_roundtrip(tmp_path) -> None:
    from scripts.train_expression import train

    out = tmp_path / "dec.pt"
    final = train(epochs=300, stop="fixed", n=1024, out=str(out))
    assert out.exists()
    assert final < 0.05  # 蒸馏分段线性解析函数，MLP 应拟合到很低
    model = load_decoder(str(out))
    channels = model.predict_channels(0.5, 0.5)
    assert CHANNELS.issubset(channels)
