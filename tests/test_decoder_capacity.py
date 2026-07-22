"""T2-C：解码器容量/深度旋钮零回归验证。

覆盖：
① 默认构造 state_dict 键集与原版逐字相同（×6）
② 加宽 hidden 后输入维不变（net[0].in_features == 2 或 dim）
③ 加宽后输出维不变（Expression→11, Physio/Prosody→3, Facs→5, text→2）
④ load_* 加载默认权重不报错
⑤ num_layers>default 时键数增加且不等于旧键集

torch 缺失时整文件跳过。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.agents.models.expression_decoder import ExpressionDecoder, load_decoder  # noqa: E402
from src.agents.models.facs_decoder import FacsDecoder, load_facs_decoder  # noqa: E402
from src.agents.models.physiology_decoder import (  # noqa: E402
    PhysiologyDecoder,
    load_physiology_decoder,
)
from src.agents.models.prosody_decoder import ProsodyDecoder, load_prosody_decoder  # noqa: E402
from src.agents.models.text_affect_regressor import (  # noqa: E402
    TEXT_FEATURE_DIM,
    TextAffectRegressor,
    load_text_affect_regressor,
)
from src.agents.models.text_affect_regressor_st import (  # noqa: E402
    ST_FEATURE_DIM,
    STTextAffectRegressor,
    load_st_text_affect_regressor,
)

# ── ① 各模型旧键集（硬编码，零回归保证） ─────────────────────────────────────

# ExpressionDecoder: hidden=32, num_layers=2
# Linear(2,32)→ReLU→Linear(32,32)→ReLU→Linear(32,11)→Sigmoid
# 索引: 0(Linear) 1(ReLU) 2(Linear) 3(ReLU) 4(Linear) 5(Sigmoid) → 带参数的是 0,2,4
EXPRESSION_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "net.4.weight",
    "net.4.bias",
}

# PhysiologyDecoder: hidden=16, num_layers=1
# Linear(2,16)→ReLU→Linear(16,3)→Sigmoid → 带参数的是 0,2
PHYSIOLOGY_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}

# ProsodyDecoder: 同 Physiology
PROSODY_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}

# FacsDecoder: 同 Physiology
FACS_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}

# TextAffectRegressor: dim=256, hidden=64, num_layers=1
# Linear(256,64)→ReLU→Linear(64,2)→Tanh → 带参数的是 0,2
TEXT_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}

# STTextAffectRegressor: dim=384, hidden=64, num_layers=1（MLP 头，同 text）
ST_TEXT_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}


# ── ① 默认构造键集不变 ────────────────────────────────────────────────────────


def test_expression_decoder_default_keys() -> None:
    m = ExpressionDecoder()
    assert set(m.state_dict().keys()) == EXPRESSION_KEYS


def test_physiology_decoder_default_keys() -> None:
    m = PhysiologyDecoder()
    assert set(m.state_dict().keys()) == PHYSIOLOGY_KEYS


def test_prosody_decoder_default_keys() -> None:
    m = ProsodyDecoder()
    assert set(m.state_dict().keys()) == PROSODY_KEYS


def test_facs_decoder_default_keys() -> None:
    m = FacsDecoder()
    assert set(m.state_dict().keys()) == FACS_KEYS


def test_text_affect_regressor_default_keys() -> None:
    m = TextAffectRegressor()
    assert set(m.state_dict().keys()) == TEXT_KEYS


def test_st_text_affect_regressor_default_keys() -> None:
    m = STTextAffectRegressor()
    assert set(m.state_dict().keys()) == ST_TEXT_KEYS


# ── ② 加宽 hidden 后输入维不变 ───────────────────────────────────────────────


def test_expression_decoder_wider_input_dim() -> None:
    m = ExpressionDecoder(hidden=64)
    assert m.net[0].in_features == 2


def test_physiology_decoder_wider_input_dim() -> None:
    m = PhysiologyDecoder(hidden=32)
    assert m.net[0].in_features == 2


def test_prosody_decoder_wider_input_dim() -> None:
    m = ProsodyDecoder(hidden=32)
    assert m.net[0].in_features == 2


def test_facs_decoder_wider_input_dim() -> None:
    m = FacsDecoder(hidden=32)
    assert m.net[0].in_features == 2


def test_text_affect_regressor_wider_input_dim() -> None:
    m = TextAffectRegressor(hidden=128)
    assert m.net[0].in_features == TEXT_FEATURE_DIM


def test_st_text_affect_regressor_wider_input_dim() -> None:
    m = STTextAffectRegressor(hidden=128)
    assert m.net[0].in_features == ST_FEATURE_DIM


# ── ③ 加宽后输出维不变 ───────────────────────────────────────────────────────


def test_expression_decoder_wider_output_dim() -> None:
    m = ExpressionDecoder(hidden=64)
    x = torch.zeros(1, 2)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 11


def test_physiology_decoder_wider_output_dim() -> None:
    m = PhysiologyDecoder(hidden=32)
    x = torch.zeros(1, 2)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 3


def test_prosody_decoder_wider_output_dim() -> None:
    m = ProsodyDecoder(hidden=32)
    x = torch.zeros(1, 2)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 3


def test_facs_decoder_wider_output_dim() -> None:
    m = FacsDecoder(hidden=32)
    x = torch.zeros(1, 2)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 5


def test_text_affect_regressor_wider_output_dim() -> None:
    m = TextAffectRegressor(hidden=128)
    x = torch.zeros(1, TEXT_FEATURE_DIM)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 2


def test_st_text_affect_regressor_wider_output_dim() -> None:
    m = STTextAffectRegressor(hidden=128)
    x = torch.zeros(1, ST_FEATURE_DIM)
    with torch.no_grad():
        out = m(x)
    assert out.shape[-1] == 2


# ── ④ load_* 加载默认权重不报错 ──────────────────────────────────────────────


def test_load_expression_decoder_roundtrip(tmp_path) -> None:
    m = ExpressionDecoder()
    p = tmp_path / "expr.pt"
    torch.save(m.state_dict(), p)
    loaded = load_decoder(str(p))
    assert set(loaded.state_dict().keys()) == EXPRESSION_KEYS


def test_load_physiology_decoder_roundtrip(tmp_path) -> None:
    m = PhysiologyDecoder()
    p = tmp_path / "physio.pt"
    torch.save(m.state_dict(), p)
    loaded = load_physiology_decoder(str(p))
    assert set(loaded.state_dict().keys()) == PHYSIOLOGY_KEYS


def test_load_prosody_decoder_roundtrip(tmp_path) -> None:
    m = ProsodyDecoder()
    p = tmp_path / "prosody.pt"
    torch.save(m.state_dict(), p)
    loaded = load_prosody_decoder(str(p))
    assert set(loaded.state_dict().keys()) == PROSODY_KEYS


def test_load_facs_decoder_roundtrip(tmp_path) -> None:
    m = FacsDecoder()
    p = tmp_path / "facs.pt"
    torch.save(m.state_dict(), p)
    loaded = load_facs_decoder(str(p))
    assert set(loaded.state_dict().keys()) == FACS_KEYS


def test_load_text_affect_regressor_roundtrip(tmp_path) -> None:
    m = TextAffectRegressor()
    p = tmp_path / "text.pt"
    torch.save(m.state_dict(), p)
    loaded = load_text_affect_regressor(str(p))
    assert set(loaded.state_dict().keys()) == TEXT_KEYS


def test_load_st_text_affect_regressor_roundtrip(tmp_path) -> None:
    m = STTextAffectRegressor()
    p = tmp_path / "st_text.pt"
    torch.save(m.state_dict(), p)
    loaded = load_st_text_affect_regressor(str(p))
    assert set(loaded.state_dict().keys()) == ST_TEXT_KEYS


# ── ⑤ num_layers > default 时键数增加且不等于旧键集 ──────────────────────────


def test_physiology_decoder_extra_layer_keys() -> None:
    """PhysiologyDecoder 默认 num_layers=1，增到 2 后键集更大且不同。"""
    m_wide = PhysiologyDecoder(num_layers=2)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(PHYSIOLOGY_KEYS)
    assert keys_wide != PHYSIOLOGY_KEYS


def test_prosody_decoder_extra_layer_keys() -> None:
    m_wide = ProsodyDecoder(num_layers=2)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(PROSODY_KEYS)
    assert keys_wide != PROSODY_KEYS


def test_facs_decoder_extra_layer_keys() -> None:
    m_wide = FacsDecoder(num_layers=2)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(FACS_KEYS)
    assert keys_wide != FACS_KEYS


def test_text_affect_regressor_extra_layer_keys() -> None:
    m_wide = TextAffectRegressor(num_layers=2)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(TEXT_KEYS)
    assert keys_wide != TEXT_KEYS


def test_st_text_affect_regressor_extra_layer_keys() -> None:
    m_wide = STTextAffectRegressor(num_layers=2)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(ST_TEXT_KEYS)
    assert keys_wide != ST_TEXT_KEYS


def test_expression_decoder_extra_layer_keys() -> None:
    """ExpressionDecoder 默认 num_layers=2，增到 3 后键集更大且不同。"""
    m_wide = ExpressionDecoder(num_layers=3)
    keys_wide = set(m_wide.state_dict().keys())
    assert len(keys_wide) > len(EXPRESSION_KEYS)
    assert keys_wide != EXPRESSION_KEYS
