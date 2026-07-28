"""三-2 RAVDESS 韵律真网络化：文件名解析 / 真音频特征提取 / 复合注入 / 训练 smoke。

用 stdlib wave 合成正弦 WAV（RAVDESS 命名）作 fixture，真实走 librosa 提特征，
无需下载 RAVDESS。torch/librosa 缺失则整文件跳过。
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("librosa")

from src.agents.datasets.ravdess import load_ravdess, parse_emotion_code  # noqa: E402
from src.agents.expression import ExpressionAgent  # noqa: E402
from src.agents.models.composite import CompositeChannelDecoder  # noqa: E402
from src.agents.models.prosody_decoder import PROSODY_DIM, ProsodyDecoder  # noqa: E402
from src.orchestration.state import AffectState  # noqa: E402

CHANNELS = {"facs_au", "text_label", "physiology", "prosody"}


def write_sine(path: Path, freq: float, *, sr: int = 16000, dur: float = 0.4) -> None:
    n = int(sr * dur)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n)
        )
        w.writeframes(frames)


@pytest.fixture
def ravdess_dir(tmp_path: Path) -> Path:
    # 文件名第 3 段为情绪码：03 高兴 / 05 愤怒 / 01 中性 / 04 悲伤
    write_sine(tmp_path / "03-01-03-01-01-01-01.wav", 300)
    write_sine(tmp_path / "03-01-05-01-01-01-02.wav", 150)
    write_sine(tmp_path / "03-01-01-01-01-01-03.wav", 220)
    write_sine(tmp_path / "03-01-04-01-01-01-04.wav", 120)
    write_sine(tmp_path / "not-ravdess.wav", 200)  # 应被跳过
    return tmp_path


def test_parse_emotion_code() -> None:
    assert parse_emotion_code("03-01-05-01-01-01-12.wav") == "05"
    assert parse_emotion_code("not-ravdess.wav") is None
    assert parse_emotion_code("03-01-99-01-01-01-01.wav") is None  # 非法情绪码


def test_load_ravdess_shapes_and_ranges(ravdess_dir: Path) -> None:
    x, y = load_ravdess(ravdess_dir)
    assert x.shape == (4, 2)  # 跳过 not-ravdess.wav
    assert y.shape == (4, PROSODY_DIM)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_load_ravdess_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ravdess(tmp_path)


def test_prosody_decoder_forward_and_predict() -> None:
    import torch

    model = ProsodyDecoder()
    with torch.no_grad():
        out = model(torch.zeros(3, 2))
    assert out.shape == (3, PROSODY_DIM)
    prosody = model.predict_prosody(0.5, 0.5)
    assert set(prosody) == {"speech_rate", "pitch", "energy"}


def test_composite_decoder_overrides_prosody_only() -> None:
    model = ProsodyDecoder()
    composite = CompositeChannelDecoder(prosody_model=model)
    channels = composite.predict_channels(0.5, 0.5)
    assert CHANNELS.issubset(channels)
    assert channels["prosody"] == model.predict_prosody(0.5, 0.5)  # 韵律来自真模型
    # zero-link Q1（2026-07-14）：专用韵律真模型出归一 [0,1] → 量纲标记 "normalized"
    assert channels["prosody_scale"] == "normalized"
    # 未注入韵律模型时回退解析占位
    fallback = CompositeChannelDecoder().predict_channels(0.5, 0.5)
    assert CHANNELS.issubset(fallback)
    assert fallback["prosody_scale"] == "ratio"  # 占位倍率口径


def test_expression_agent_with_composite_preserves_contract() -> None:
    agent = ExpressionAgent(decoder=CompositeChannelDecoder(prosody_model=ProsodyDecoder()))
    expr = agent(AffectState(affect_sample=(0.8, 0.7)))["expression"]
    assert expr["valence_arousal"] is not None
    for head in ("spontaneous", "voluntary"):
        assert CHANNELS.issubset(expr[head]), head


def test_train_prosody_smoke(ravdess_dir: Path, tmp_path: Path) -> None:
    from scripts.train_prosody import train

    out = tmp_path / "prosody.pt"
    final = train(str(ravdess_dir), epochs=100, stop="fixed", out=str(out))
    assert out.exists()
    assert math.isfinite(final)
    assert final < 0.2  # 仅 4 样本，应能快速拟合

    # provenance sidecar 与权重同产（旁挂 json，不改 .pt 格式）
    import json

    from scripts._train_common import provenance_path

    rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
    assert rec["script"] == "scripts/train_prosody.py"
    assert rec["training"]["epochs_ran"] == 100
    assert rec["data"]["kind"] == "directory"  # RAVDESS 根是目录，不哈希
