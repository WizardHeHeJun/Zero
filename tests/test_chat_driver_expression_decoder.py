"""composite 接入 live 运行时 env 工厂（阶段 36 follow-up）单测。

覆盖 `_build_expression_decoder` 与 `build_chat_driver` 的表情解码器装配：
- 默认关零回归：ZERO_FACS_MODEL_PATH 未设 → None → ExpressionAgent 占位路径（不需 torch）。
- env 门控开：真权重加载 → CompositeChannelDecoder 构造，facs_au 键集被真模型接管。
- 三系数贯通：ZERO_FACS_K_AROUSAL / K_COPING / RESIDUAL_ALPHA 构造期读入；未设=构造函数默认。
- fail-fast：权重形状与 facs_extended 不配对（RuntimeError）、residual_alpha 越界（ValueError）。
- 同源契约：工厂端 ZERO_FACS_EXTENDED 同时驱动 state 门控与模型 extended（键集对齐）。

torch 侧用 importorskip（同 test_decoder_capacity 先例）；权重用随机初始化 state_dict——
测的是接线与形状契约，非表情精度（真权重回归见 scripts/train_facs 侧）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.orchestration.chat_driver import _build_expression_decoder, build_chat_driver

_FACS_ENVS = (
    "ZERO_FACS_MODEL_PATH",
    "ZERO_FACS_K_AROUSAL",
    "ZERO_FACS_K_COPING",
    "ZERO_FACS_RESIDUAL_ALPHA",
    "ZERO_FACS_EXTENDED",
    # 独立通道门控（zero-link T4）：清掉防真 .env 的 prosody 权重污染 FACS 用例（会翻 normalized）。
    "ZERO_PROSODY_MODEL_PATH",
    # 独立通道门控（zero-link physiology 2026-07-23）：同理清 physiology 权重防污染。
    "ZERO_PHYSIOLOGY_MODEL_PATH",
    # 占位口径门控（zero-link 任务② 2026-07-23）：清防真 .env 开着污染零回归/接线用例。
    "ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER",
)


def _clear_facs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """钉死 FACS/prosody 解码器 env 全未设（不受跑测环境 .env 污染），保测试独立。"""
    for name in _FACS_ENVS:
        monkeypatch.delenv(name, raising=False)


def _save_ext_weights(tmp_path: Path) -> str:
    """存一份形状正确的 11-AU 扩展权重（随机初始化即可——测接线与形状契约，非精度）。"""
    torch = pytest.importorskip("torch")
    from src.agents.models.facs_decoder import FacsDecoder

    model = FacsDecoder(extended=True)
    path = tmp_path / "facs_decoder_ext.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _save_prosody_weights(tmp_path: Path) -> str:
    """存一份默认架构（hidden=16/num_layers=1）的 ProsodyDecoder 权重（随机初始化即可·测接线）。"""
    torch = pytest.importorskip("torch")
    from src.agents.models.prosody_decoder import ProsodyDecoder

    model = ProsodyDecoder()
    path = tmp_path / "prosody_decoder.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _save_physiology_weights(tmp_path: Path) -> str:
    """存一份默认架构（hidden=16/num_layers=1）的 PhysiologyDecoder 权重（随机初始化·测接线）。"""
    torch = pytest.importorskip("torch")
    from src.agents.models.physiology_decoder import PhysiologyDecoder

    model = PhysiologyDecoder()
    path = tmp_path / "physiology_decoder.pt"
    torch.save(model.state_dict(), path)
    return str(path)


class _CaptureSession:
    """捕获 build_chat_driver 传给 ConversationSession 的 kwargs（不真建图，无后端依赖）。

    kwargs 存实例变量、经 `driver.session` 取（W1 整改）：不落类变量共享态，测试间零顺序
    依赖；工厂若在 ConversationSession(...) 之前抛异常也不会留下旧值误判。
    """

    def __init__(self, **kwargs: Any) -> None:
        self.captured = kwargs


def test_unset_model_path_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认（未设 ZERO_FACS_MODEL_PATH）→ None → 占位路径（零回归；此路径不 import torch）。"""
    _clear_facs_env(monkeypatch)
    assert _build_expression_decoder(False) is None
    assert _build_expression_decoder(True) is None


def test_factory_defaults_to_none_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_chat_driver 默认把 expression_decoder=None 传给 ConversationSession（零回归）。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setattr("src.orchestration.chat_driver.ConversationSession", _CaptureSession)
    driver = build_chat_driver(thread="test-facs-none")
    session = driver.session
    assert isinstance(session, _CaptureSession)
    assert session.captured["expression_decoder"] is None


def test_env_gate_constructs_composite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """设 ZERO_FACS_MODEL_PATH → 真权重加载进 CompositeChannelDecoder；三系数 env 贯通。"""
    _clear_facs_env(monkeypatch)
    weights = _save_ext_weights(tmp_path)
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.facs_decoder import FACS_KEYS_EXT, FacsDecoder

    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", weights)
    monkeypatch.setenv("ZERO_FACS_K_AROUSAL", "2.0")
    monkeypatch.setenv("ZERO_FACS_K_COPING", "0.7")
    monkeypatch.setenv("ZERO_FACS_RESIDUAL_ALPHA", "0.5")
    decoder = _build_expression_decoder(True)
    assert isinstance(decoder, CompositeChannelDecoder)
    assert isinstance(decoder.facs_model, FacsDecoder)
    assert decoder.facs_model.extended is True
    assert decoder.facs_extended is True
    assert decoder.k_arousal == pytest.approx(2.0)
    assert decoder.k_coping == pytest.approx(0.7)
    assert decoder.residual_alpha == pytest.approx(0.5)
    # 真模型接管 facs_au：键集为 11-AU 扩展集合（其余通道回退占位，text_label 仍在）
    channels = decoder.predict_channels(0.5, 0.5)
    assert set(channels["facs_au"]) == set(FACS_KEYS_EXT)


def test_coefficient_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """三系数 env 未设 → 构造函数默认 1.5/1.2/1.0（幅度零回归）。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", _save_ext_weights(tmp_path))
    decoder = _build_expression_decoder(True)
    assert decoder is not None
    assert decoder.k_arousal == pytest.approx(1.5)
    assert decoder.k_coping == pytest.approx(1.2)
    assert decoder.residual_alpha == pytest.approx(1.0)


def test_shape_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """11-AU 权重 × facs_extended=False（5 维模型）→ load_state_dict 即抛，不静默回退占位。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", _save_ext_weights(tmp_path))
    with pytest.raises(RuntimeError):
        _build_expression_decoder(False)


def test_missing_weight_file_fails_fast_with_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """路径打错/文件缺失 → RuntimeError 指向 ZERO_FACS_MODEL_PATH（W3：非 torch 原始堆栈裸抛）。"""
    pytest.importorskip("torch")
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", str(tmp_path / "nonexistent.pt"))
    with pytest.raises(RuntimeError, match="ZERO_FACS_MODEL_PATH"):
        _build_expression_decoder(True)


def test_invalid_residual_alpha_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ZERO_FACS_RESIDUAL_ALPHA 越界 → CompositeChannelDecoder 构造抛 ValueError（fail-fast）。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", _save_ext_weights(tmp_path))
    monkeypatch.setenv("ZERO_FACS_RESIDUAL_ALPHA", "1.5")
    with pytest.raises(ValueError, match="residual_alpha"):
        _build_expression_decoder(True)


def test_factory_extended_same_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """工厂端到端：同一 ZERO_FACS_EXTENDED 驱动 state 门控与模型 extended（键集对齐契约）。"""
    _clear_facs_env(monkeypatch)
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.facs_decoder import FacsDecoder

    monkeypatch.setenv("ZERO_FACS_MODEL_PATH", _save_ext_weights(tmp_path))
    monkeypatch.setenv("ZERO_FACS_EXTENDED", "true")
    monkeypatch.setattr("src.orchestration.chat_driver.ConversationSession", _CaptureSession)
    driver = build_chat_driver(thread="test-facs-composite")
    session = driver.session
    assert isinstance(session, _CaptureSession)
    decoder = session.captured["expression_decoder"]
    assert isinstance(decoder, CompositeChannelDecoder)
    assert decoder.facs_extended is True
    assert session.captured["facs_extended"] is True
    assert isinstance(decoder.facs_model, FacsDecoder)
    assert decoder.facs_model.extended is True


def test_prosody_env_gate_wires_prosody_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """设 ZERO_PROSODY_MODEL_PATH → 真 ProsodyDecoder 注入，prosody_scale 翻 normalized。"""
    _clear_facs_env(monkeypatch)
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.prosody_decoder import ProsodyDecoder

    monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", _save_prosody_weights(tmp_path))
    decoder = _build_expression_decoder(False)
    assert isinstance(decoder, CompositeChannelDecoder)
    assert isinstance(decoder.prosody_model, ProsodyDecoder)
    channels = decoder.predict_channels(0.5, 0.5)
    assert channels["prosody_scale"] == "normalized"
    assert all(0.0 <= v <= 1.0 for v in channels["prosody"].values())


def test_prosody_only_no_facs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """通道独立门控：只设 prosody（facs 未设）→ decoder 非 None、facs_model=None。"""
    _clear_facs_env(monkeypatch)
    from src.agents.models.composite import CompositeChannelDecoder

    monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", _save_prosody_weights(tmp_path))
    decoder = _build_expression_decoder(False)
    assert isinstance(decoder, CompositeChannelDecoder)
    assert decoder.facs_model is None  # facs 未设 → facs_au 仍走解析占位（零回归）
    assert decoder.prosody_model is not None
    assert decoder.predict_channels(0.5, 0.5)["prosody_scale"] == "normalized"


def test_missing_prosody_weight_fails_fast_with_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """prosody 路径打错/缺失 → RuntimeError 指向 ZERO_PROSODY_MODEL_PATH。"""
    pytest.importorskip("torch")
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", str(tmp_path / "nonexistent.pt"))
    with pytest.raises(RuntimeError, match="ZERO_PROSODY_MODEL_PATH"):
        _build_expression_decoder(False)


def test_physiology_env_gate_wires_physiology_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """设 ZERO_PHYSIOLOGY_MODEL_PATH → 真 PhysiologyDecoder 注入，physiology 出 WESAD 真信号量纲
    （sc μS[0,20]、temperature_c °C[30,40]、含 temperature_c 键·无 pupil_mm）。"""
    _clear_facs_env(monkeypatch)
    from src.agents.models.composite import CompositeChannelDecoder
    from src.agents.models.physiology_decoder import PhysiologyDecoder

    monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", _save_physiology_weights(tmp_path))
    decoder = _build_expression_decoder(False)
    assert isinstance(decoder, CompositeChannelDecoder)
    assert isinstance(decoder.physiology_model, PhysiologyDecoder)
    physio = decoder.predict_channels(0.5, 0.5)["physiology"]
    # canonical=WESAD：真 decoder 出 {hr,sc(μS),temperature_c}，替换占位的 pupil_mm。
    assert set(physio) == {"heart_rate_bpm", "skin_conductance", "temperature_c"}
    assert 50.0 <= physio["heart_rate_bpm"] <= 120.0  # sigmoid 反归一域
    assert 0.0 <= physio["skin_conductance"] <= 20.0  # μS
    assert 30.0 <= physio["temperature_c"] <= 40.0  # °C


def test_physiology_only_no_facs_no_prosody(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """通道独立门控：只设 physiology（facs/prosody 未设）→ decoder 非 None、physiology 接真模型。"""
    _clear_facs_env(monkeypatch)
    from src.agents.models.composite import CompositeChannelDecoder

    monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", _save_physiology_weights(tmp_path))
    decoder = _build_expression_decoder(False)
    assert isinstance(decoder, CompositeChannelDecoder)
    assert decoder.facs_model is None  # facs 未设 → facs_au 仍走解析占位（零回归）
    assert decoder.prosody_model is None  # prosody 未设 → prosody 仍走解析占位、scale=ratio
    assert decoder.physiology_model is not None
    # ratio = decode_channels 占位约定（affect_math），非 physiology decoder 产出；
    # prosody 未注入 → 不翻 normalized，佐证 physiology 门控不越界影响 prosody。
    assert decoder.predict_channels(0.5, 0.5)["prosody_scale"] == "ratio"


def test_missing_physiology_weight_fails_fast_with_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """physiology 路径打错/缺失 → RuntimeError 指向 ZERO_PHYSIOLOGY_MODEL_PATH。"""
    pytest.importorskip("torch")
    _clear_facs_env(monkeypatch)
    monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", str(tmp_path / "nonexistent.pt"))
    with pytest.raises(RuntimeError, match="ZERO_PHYSIOLOGY_MODEL_PATH"):
        _build_expression_decoder(False)


def test_canonical_physiology_env_propagates_to_session_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """集成回归（code-reviewer BLOCK-1）：`ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER=true` 须经
    build_chat_driver 贯通到 ConversationSession/SessionConfig（否则 chat 占位路径 A[decoder=None]
    读到的 state.canonical_physiology 恒 False、门开失效）。经真 build_chat_driver 捕获
    SessionConfig kwargs（非直设 state）——正是"漏传给 ConversationSession"bug 的探测点。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setattr("src.orchestration.chat_driver.ConversationSession", _CaptureSession)
    monkeypatch.setenv("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", "true")
    driver = build_chat_driver(thread="test-canonical-physiology-on")
    session = driver.session
    assert isinstance(session, _CaptureSession)
    assert session.captured["canonical_physiology"] is True


def test_canonical_physiology_default_off_zero_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认（未设 env）→ SessionConfig.canonical_physiology=False（逐字零回归）。"""
    _clear_facs_env(monkeypatch)
    monkeypatch.setattr("src.orchestration.chat_driver.ConversationSession", _CaptureSession)
    driver = build_chat_driver(thread="test-canonical-physiology-off")
    session = driver.session
    assert isinstance(session, _CaptureSession)
    assert session.captured["canonical_physiology"] is False
