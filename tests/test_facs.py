"""三-4 FACS 表情真网络化：CSV loader / 解码器 / 复合注入 / 训练 smoke。

合成最小 FACS 标注 CSV 作 fixture，无需 EULA 数据集。torch 缺失则整文件跳过。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.agents.datasets.facs_csv import load_facs_csv  # noqa: E402
from src.agents.expression import ExpressionAgent  # noqa: E402
from src.agents.models.composite import CompositeChannelDecoder  # noqa: E402
from src.agents.models.facs_decoder import FACS_DIM, FACS_KEYS, FacsDecoder  # noqa: E402
from src.orchestration.state import AffectState  # noqa: E402

CHANNELS = {"facs_au", "text_label", "physiology", "prosody"}


def make_facs_csv(path: Path) -> None:
    rows = [
        {
            "valence": 0.8,
            "arousal": 0.5,
            "AU04": 0.0,
            "AU06": 0.5,
            "AU12": 0.8,
            "AU15": 0.0,
            "intensity": 0.5,
        },
        {
            "valence": -0.7,
            "arousal": 0.3,
            "AU04": 0.4,
            "AU06": 0.0,
            "AU12": 0.0,
            "AU15": 0.7,
            "intensity": 0.3,
        },
        {
            "valence": 0.0,
            "arousal": 0.0,
            "AU04": 0.0,
            "AU06": 0.0,
            "AU12": 0.0,
            "AU15": 0.0,
            "intensity": 0.0,
        },
        {
            "valence": 0.3,
            "arousal": 0.9,
            "AU04": 0.1,
            "AU06": 0.3,
            "AU12": 0.4,
            "AU15": 0.1,
            "intensity": 0.9,
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["valence", "arousal", *FACS_KEYS])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def facs_csv(tmp_path: Path) -> Path:
    path = tmp_path / "labels.csv"
    make_facs_csv(path)
    return path


def test_load_facs_csv_shapes_and_ranges(facs_csv: Path) -> None:
    x, y = load_facs_csv(facs_csv)
    assert x.shape == (4, 2)
    assert y.shape == (4, FACS_DIM)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_load_facs_csv_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["valence", "arousal", *FACS_KEYS]).writeheader()
    with pytest.raises(ValueError):
        load_facs_csv(path)


def test_facs_decoder_forward_and_predict() -> None:
    import torch

    model = FacsDecoder()
    with torch.no_grad():
        out = model(torch.zeros(3, 2))
    assert out.shape == (3, FACS_DIM)
    facs = model.predict_facs(0.5, 0.5)
    assert set(facs) == set(FACS_KEYS)


def test_composite_overrides_facs_channel() -> None:
    composite = CompositeChannelDecoder(facs_model=FacsDecoder())
    channels = composite.predict_channels(0.5, 0.5)
    assert CHANNELS.issubset(channels)
    assert set(channels["facs_au"]) == set(FACS_KEYS)


def test_expression_agent_with_facs_composite_preserves_contract() -> None:
    agent = ExpressionAgent(decoder=CompositeChannelDecoder(facs_model=FacsDecoder()))
    expr = agent(AffectState(affect_sample=(0.8, 0.5)))["expression"]
    assert expr["valence_arousal"] is not None
    for head in ("spontaneous", "voluntary"):
        assert CHANNELS.issubset(expr[head]), head


def test_train_facs_smoke(facs_csv: Path, tmp_path: Path) -> None:
    from scripts.train_facs import train

    out = tmp_path / "facs.pt"
    final = train(str(facs_csv), epochs=200, out=str(out))
    assert out.exists()
    assert math.isfinite(final)
    assert final < 0.1  # 仅 4 样本，应快速拟合
