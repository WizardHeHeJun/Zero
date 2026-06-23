"""三-3 WESAD 生理真网络化：condition→(v,a) / 真信号特征 / 复合注入 / 训练 smoke。

合成符合 WESAD 结构的 pkl（含周期尖峰 ECG）作 fixture，真实走 scipy 心率检测，
无需下载 WESAD。torch/numpy/scipy 缺失则整文件跳过。
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402

from src.agents.datasets.wesad import CHEST_FS, load_wesad  # noqa: E402
from src.agents.expression import ExpressionAgent  # noqa: E402
from src.agents.models.composite import CompositeChannelDecoder  # noqa: E402
from src.agents.models.physiology_decoder import (  # noqa: E402
    PHYSIOLOGY_DIM,
    PhysiologyDecoder,
)
from src.agents.models.prosody_decoder import ProsodyDecoder  # noqa: E402
from src.orchestration.state import AffectState  # noqa: E402

CHANNELS = {"facs_au", "text_label", "physiology", "prosody"}


def make_wesad_pkl(path: Path, *, seconds: int = 5) -> None:
    """合成一份最小 WESAD pkl：4 个 condition 各一段，ECG 为周期尖峰。"""
    fs = CHEST_FS
    ecg_parts, eda_parts, temp_parts, label_parts = [], [], [], []
    for condition, bpm in [(1, 60), (2, 90), (3, 75), (4, 55)]:
        n = fs * seconds
        beat = int(fs * 60 / bpm)
        sig = np.zeros(n)
        sig[beat // 2 :: beat] = 3.0  # 尖峰不落在 index 0，确保可检出
        ecg_parts.append(sig)
        eda_parts.append(np.full(n, float(condition) * 2.0))
        temp_parts.append(np.full(n, 33.0))
        label_parts.append(np.full(n, condition))

    data = {
        "signal": {
            "chest": {
                "ECG": np.concatenate(ecg_parts).reshape(-1, 1),
                "EDA": np.concatenate(eda_parts).reshape(-1, 1),
                "Temp": np.concatenate(temp_parts).reshape(-1, 1),
            }
        },
        "label": np.concatenate(label_parts),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


@pytest.fixture
def wesad_dir(tmp_path: Path) -> Path:
    subject = tmp_path / "S2"
    subject.mkdir()
    make_wesad_pkl(subject / "S2.pkl", seconds=5)
    return tmp_path


def test_load_wesad_shapes_and_ranges(wesad_dir: Path) -> None:
    x, y = load_wesad(wesad_dir, window_seconds=2)  # 5s 段 → 每 condition 2 窗
    assert x.shape[0] > 0
    assert x.shape[1] == 2
    assert y.shape[1] == PHYSIOLOGY_DIM
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_load_wesad_window_too_large_raises(wesad_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_wesad(wesad_dir, window_seconds=999)


def test_estimate_heart_rate_on_synthetic() -> None:
    from src.agents.datasets.wesad import estimate_heart_rate

    fs = CHEST_FS
    n = fs * 4
    beat = int(fs * 60 / 72)  # 72 bpm
    sig = np.zeros(n)
    sig[beat // 2 :: beat] = 3.0
    hr = estimate_heart_rate(sig, fs=fs)
    assert math.isclose(hr, 72.0, abs_tol=3.0)


def test_physiology_decoder_forward_and_predict() -> None:
    import torch

    model = PhysiologyDecoder()
    with torch.no_grad():
        out = model(torch.zeros(3, 2))
    assert out.shape == (3, PHYSIOLOGY_DIM)
    physio = model.predict_physiology(0.5, 0.5)
    assert set(physio) == {"heart_rate_bpm", "skin_conductance", "temperature_c"}


def test_composite_overrides_both_channels() -> None:
    composite = CompositeChannelDecoder(
        prosody_model=ProsodyDecoder(), physiology_model=PhysiologyDecoder()
    )
    channels = composite.predict_channels(0.5, 0.5)
    assert CHANNELS.issubset(channels)
    # 生理与韵律都来自真模型
    assert set(channels["physiology"]) == {"heart_rate_bpm", "skin_conductance", "temperature_c"}
    assert set(channels["prosody"]) == {"speech_rate", "pitch", "energy"}


def test_expression_agent_with_physiology_composite_preserves_contract() -> None:
    agent = ExpressionAgent(decoder=CompositeChannelDecoder(physiology_model=PhysiologyDecoder()))
    expr = agent(AffectState(affect_sample=(-0.6, 0.7)))["expression"]
    assert expr["valence_arousal"] is not None
    for head in ("spontaneous", "voluntary"):
        assert CHANNELS.issubset(expr[head]), head


def test_train_physiology_smoke(wesad_dir: Path, tmp_path: Path) -> None:
    from scripts.train_physiology import train

    out = tmp_path / "physio.pt"
    final = train(str(wesad_dir), epochs=100, window_seconds=2, out=str(out))
    assert out.exists()
    assert math.isfinite(final)
    assert final < 0.2
