"""句向量版文本→V-A 输入侧：共用 loader 解析 / ST 回归器 / 训练 smoke。

需要 torch + sentence-transformers，缺任一则整文件跳过（首次会下载 MiniLM 权重）。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from src.agents.datasets.emobank import read_emobank_rows  # noqa: E402
from src.agents.datasets.emobank_st import load_emobank_embeddings  # noqa: E402
from src.agents.models.text_affect_regressor_st import (  # noqa: E402
    ST_FEATURE_DIM,
    STTextAffectRegressor,
)


def make_emobank_csv(path: Path) -> None:
    rows = [
        {
            "id": "1",
            "split": "train",
            "V": "4.5",
            "A": "3.8",
            "D": "3.0",
            "text": "wonderful joyful day",
        },
        {
            "id": "2",
            "split": "train",
            "V": "1.5",
            "A": "4.2",
            "D": "3.0",
            "text": "terrible angry news",
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
            "text": "feeling sad and tired",
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


def test_read_emobank_rows_parses_and_normalizes(emobank_csv: Path) -> None:
    texts, ys = read_emobank_rows(emobank_csv)
    assert len(texts) == len(ys) == 4
    assert all(-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0 for v, a in ys)


def test_read_emobank_rows_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["id", "split", "V", "A", "D", "text"]).writeheader()
    with pytest.raises(ValueError):
        read_emobank_rows(path)


def test_st_regressor_forward_zeros() -> None:
    import torch

    model = STTextAffectRegressor()
    with torch.no_grad():
        out = model(torch.zeros(2, ST_FEATURE_DIM))
    assert out.shape == (2, 2)


def test_load_embeddings_and_predict(emobank_csv: Path) -> None:
    x, y = load_emobank_embeddings(emobank_csv)
    assert x.shape == (4, ST_FEATURE_DIM)
    assert y.shape == (4, 2)
    model = STTextAffectRegressor(dim=x.shape[1])
    v, a = model.predict_affect("a neutral sentence")
    assert -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0


def test_train_st_smoke(emobank_csv: Path, tmp_path: Path) -> None:
    from scripts.train_text_affect_st import train

    out = tmp_path / "st.pt"
    final = train(str(emobank_csv), epochs=200, out=str(out))
    assert out.exists()
    assert math.isfinite(final)

    # provenance sidecar 与权重同产（旁挂 json，不改 .pt 格式）
    import json

    from scripts._train_common import provenance_path

    rec = json.loads(provenance_path(out).read_text(encoding="utf-8"))
    assert rec["script"] == "scripts/train_text_affect_st.py"
    assert rec["training"]["epochs_ran"] == 200
    assert rec["model"]["encoder"] and rec["model"]["dim"] > 0  # 换编码器即换表征，须落账
    assert rec["data"]["kind"] == "file"
