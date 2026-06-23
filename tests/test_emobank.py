"""三-5 EmoBank 文本→V-A 输入侧：CSV loader / 哈希特征 / 回归器 / 训练 smoke。

合成最小 EmoBank 风格 CSV 作 fixture。torch 缺失则整文件跳过。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.agents.datasets.emobank import load_emobank  # noqa: E402
from src.agents.models.text_affect_regressor import (  # noqa: E402
    TEXT_FEATURE_DIM,
    TextAffectRegressor,
    hash_features,
)


def make_emobank_csv(path: Path) -> None:
    rows = [
        {
            "id": "1",
            "split": "train",
            "V": "4.5",
            "A": "3.8",
            "D": "3.0",
            "text": "what a wonderful joyful day",
        },
        {
            "id": "2",
            "split": "train",
            "V": "1.5",
            "A": "4.2",
            "D": "3.0",
            "text": "terrible angry awful news",
        },
        {
            "id": "3",
            "split": "train",
            "V": "3.0",
            "A": "3.0",
            "D": "3.0",
            "text": "the meeting is at noon",
        },
        {
            "id": "4",
            "split": "train",
            "V": "2.0",
            "A": "1.8",
            "D": "3.0",
            "text": "feeling sad and tired quietly",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "V", "A", "D", "text"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def emobank_csv(tmp_path: Path) -> Path:
    path = tmp_path / "emobank.csv"
    make_emobank_csv(path)
    return path


def test_hash_features_deterministic_and_dim() -> None:
    a = hash_features("hello world", dim=64)
    b = hash_features("hello world", dim=64)
    assert a == b  # 稳定哈希，跨调用确定
    assert len(a) == 64
    assert math.isclose(sum(a), 1.0, abs_tol=1e-6)  # 归一化


def test_load_emobank_shapes_and_ranges(emobank_csv: Path) -> None:
    x, y = load_emobank(emobank_csv)
    assert x.shape == (4, TEXT_FEATURE_DIM)
    assert y.shape == (4, 2)
    assert float(y.min()) >= -1.0 and float(y.max()) <= 1.0


def test_load_emobank_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["id", "split", "V", "A", "D", "text"]).writeheader()
    with pytest.raises(ValueError):
        load_emobank(path)


def test_regressor_forward_and_predict_range() -> None:
    import torch

    model = TextAffectRegressor()
    with torch.no_grad():
        out = model(torch.zeros(2, TEXT_FEATURE_DIM))
    assert out.shape == (2, 2)
    v, a = model.predict_affect("a neutral sentence")
    assert -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0


def test_train_text_affect_smoke(emobank_csv: Path, tmp_path: Path) -> None:
    from scripts.train_text_affect import train

    out = tmp_path / "text.pt"
    final = train(str(emobank_csv), epochs=300, out=str(out))
    assert out.exists()
    assert math.isfinite(final)
    assert final < 0.2  # 仅 4 样本，应快速拟合
