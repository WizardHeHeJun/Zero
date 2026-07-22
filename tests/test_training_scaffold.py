"""批 3 训练/标签升级脚手架单测。

覆盖：
  ① include_d=False loader 零回归（shape (n,2)）vs include_d=True（shape (n,3)、第三列∈[-1,1]）
  ② STTextAffectRegressor(output_dim=2) 默认 state_dict 键集与批 2 后版本一致
  ③ output_dim=1 前向 out.shape==(1,1)
  ④ finetune_encoder=False 时 state_dict 不含 encoder 键

torch 缺失则整文件跳过；finetune_encoder=True 的 GPU 项在无 CUDA 时 skip。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.agents.datasets.emobank import read_emobank_rows  # noqa: E402
from src.agents.datasets.emobank_st import load_emobank_embeddings  # noqa: E402
from src.agents.models.text_affect_regressor_st import (  # noqa: E402
    ST_FEATURE_DIM,
    STTextAffectRegressor,
)

# ---------------------------------------------------------------------------
# Fixture：最小 EmoBank 风格 CSV（4 行，含 V/A/D/text）
# ---------------------------------------------------------------------------


def _make_emobank_csv(path: Path) -> None:
    rows = [
        {"id": "1", "split": "train", "V": "4.5", "A": "3.8", "D": "4.0", "text": "joyful day"},
        {"id": "2", "split": "train", "V": "1.5", "A": "4.2", "D": "2.0", "text": "angry news"},
        {"id": "3", "split": "train", "V": "3.0", "A": "3.0", "D": "3.0", "text": "noon meeting"},
        {"id": "4", "split": "train", "V": "2.0", "A": "1.8", "D": "1.5", "text": "sad tired"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "V", "A", "D", "text"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def emobank_csv(tmp_path: Path) -> Path:
    path = tmp_path / "emobank.csv"
    _make_emobank_csv(path)
    return path


# ---------------------------------------------------------------------------
# ① include_d 零回归 / 含 D 分支
# ---------------------------------------------------------------------------


def test_read_emobank_rows_default_shape(emobank_csv: Path) -> None:
    """include_d=False（默认）：Y shape=(n,2)，各元素∈[-1,1]。"""
    texts, ys = read_emobank_rows(emobank_csv)
    assert len(ys) == 4
    for row in ys:
        assert len(row) == 2, "默认只应含 V/A 两列"
        assert all(-1.0 <= v <= 1.0 for v in row)


def test_read_emobank_rows_include_d_shape_and_range(emobank_csv: Path) -> None:
    """include_d=True：Y shape=(n,3)，第三列∈[-1,1]。"""
    texts, ys = read_emobank_rows(emobank_csv, include_d=True)
    assert len(ys) == 4
    for row in ys:
        assert len(row) == 3, "include_d=True 应含 V/A/D 三列"
        assert all(-1.0 <= v <= 1.0 for v in row), f"归一化后超出 [-1,1]：{row}"


def test_load_emobank_embeddings_default_y_shape(emobank_csv: Path) -> None:
    """load_emobank_embeddings 默认 include_d=False，Y.shape[-1]==2（零回归）。"""
    pytest.importorskip("sentence_transformers")
    _x, y = load_emobank_embeddings(emobank_csv)
    assert y.shape == (4, 2)


def test_load_emobank_embeddings_include_d_y_shape(emobank_csv: Path) -> None:
    """load_emobank_embeddings(include_d=True)：Y.shape[-1]==3，值∈[-1,1]。"""
    pytest.importorskip("sentence_transformers")
    _x, y = load_emobank_embeddings(emobank_csv, include_d=True)
    assert y.shape == (4, 3)
    assert float(y.min()) >= -1.0 and float(y.max()) <= 1.0


# ---------------------------------------------------------------------------
# ② 默认 state_dict 键集与批 2 后版本一致
# ---------------------------------------------------------------------------

# 批 2 后版本（num_layers=1, output_dim=2, finetune_encoder=False）的期望键集：
# Sequential = [Linear(dim,hidden), ReLU, Linear(hidden,2), Tanh]
# 索引 0=Linear, 1=ReLU(无参), 2=Linear, 3=Tanh(无参)
_BATCH2_KEYS = frozenset(["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"])


def test_default_state_dict_keys_match_batch2() -> None:
    """output_dim=2 默认构造的 state_dict 键集与批 2 后版本逐字相同。"""
    model = STTextAffectRegressor()
    keys = frozenset(model.state_dict().keys())
    assert keys == _BATCH2_KEYS, f"键集偏离批 2：got={keys}, expected={_BATCH2_KEYS}"


# ---------------------------------------------------------------------------
# ③ output_dim=1 前向 out.shape==(1,1)
# ---------------------------------------------------------------------------


def test_output_dim_1_forward_shape() -> None:
    """output_dim=1 时单样本前向输出 shape==(1,1)。"""
    model = STTextAffectRegressor(output_dim=1)
    with torch.no_grad():
        out = model(torch.zeros(1, ST_FEATURE_DIM))
    assert out.shape == (1, 1), f"期望 (1,1)，得到 {out.shape}"


def test_output_dim_3_forward_shape() -> None:
    """output_dim=3 时批量前向输出 shape==(2,3)。"""
    model = STTextAffectRegressor(output_dim=3)
    with torch.no_grad():
        out = model(torch.zeros(2, ST_FEATURE_DIM))
    assert out.shape == (2, 3), f"期望 (2,3)，得到 {out.shape}"


# ---------------------------------------------------------------------------
# ④ finetune_encoder=False 时 state_dict 不含 encoder 键
# ---------------------------------------------------------------------------


def test_frozen_encoder_no_encoder_keys_in_state_dict() -> None:
    """finetune_encoder=False（默认）：state_dict 不含任何 encoder_module 键。"""
    model = STTextAffectRegressor(finetune_encoder=False)
    keys = list(model.state_dict().keys())
    encoder_keys = [k for k in keys if "encoder" in k]
    assert not encoder_keys, f"冻结模式不应含 encoder 键，但发现：{encoder_keys}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="finetune_encoder=True 需 GPU（本机无 CUDA，跳过）",
)
def test_finetune_encoder_includes_encoder_params() -> None:
    """finetune_encoder=True 时 state_dict 含 encoder_module 键（端到端可训练）。"""
    pytest.importorskip("sentence_transformers")
    model = STTextAffectRegressor(finetune_encoder=True)
    keys = list(model.state_dict().keys())
    encoder_keys = [k for k in keys if k.startswith("encoder_module")]
    assert encoder_keys, "finetune_encoder=True 时应持有 encoder_module 参数"


def test_finetune_forward_texts_flows_gradient_to_encoder() -> None:
    """W4 修复 correctness：forward_texts 使梯度真正流到编码器参数（CPU 可验，非 GPU-gated）。

    旧实现 finetune_encoder=True 时 forward(预计算张量) 不经 encoder → 编码器拿不到梯度、
    微调是静默 no-op。修复后训练走 forward_texts（文本→encoder→头），此测试跑一步 backward
    并断言至少一个 encoder_module 参数 .grad 非 None（梯度已流入）。
    """
    pytest.importorskip("sentence_transformers")
    model = STTextAffectRegressor(finetune_encoder=True)
    model.train()
    pred = model.forward_texts(["a good day", "a bad day"])
    assert pred.shape == (2, 2)
    target = torch.zeros(2, 2)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    enc_grads = [
        p.grad
        for p in model.encoder_module.parameters()  # type: ignore[union-attr]
        if p.grad is not None
    ]
    assert enc_grads, "forward_texts 后编码器参数应收到梯度（否则微调无效）"
