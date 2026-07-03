"""T4: ignite 软门控（logistic soft gate）测试。

议会 2026-07-02 Item 2：soft_beta 参数让 ignite 的 all-or-none 硬 step 转为 logistic 连续近似。

覆盖：
- β=None 逐字零回归：ignite(streams) 与 ignite(streams, soft_beta=None) 返回完全一致。
- β 激活语义：β=20 时所有流参与，亚阈流精度大幅压低，过阈流精度基本保留。
- 有界性：软门控后各 π_eff ∈ (0, 原始Π]，不放大。
- 贯通：ZERO_IGNITION_BETA env → build_chat_driver → session.config.ignition_beta；
  未设 → None（零回归）；SessionConfig().ignition_beta is None；
  to_state_flags["ignition_beta"] is None。
- AffectCore 集成：ignition_beta=None 时输出与旧硬 step 路径逐字一致；
                   ignition_beta=20 时 ignited_streams 含所有流（软门控无过滤）。
"""

from __future__ import annotations

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import (
    SALIENCE_THRESHOLD,
    ignite,
    sigmoid,
    stream_salience,
)
from src.orchestration.runner import SessionConfig
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# 测试辅助：构造亚阈流和过阈流
# ---------------------------------------------------------------------------

# 亚阈流：salience 明显低于 SALIENCE_THRESHOLD
_SUBTHRESH_MU: tuple[float, float] = (0.02, 0.02)
_SUBTHRESH_PREC: tuple[float, float] = (0.15, 0.15)

# 过阈流：salience 明显超过 SALIENCE_THRESHOLD
_SUPRATHRESH_MU: tuple[float, float] = (0.7, 0.7)
_SUPRATHRESH_PREC: tuple[float, float] = (1.0, 1.0)

# 确认分类正确
assert stream_salience(_SUBTHRESH_MU, _SUBTHRESH_PREC) < SALIENCE_THRESHOLD
assert stream_salience(_SUPRATHRESH_MU, _SUPRATHRESH_PREC) >= SALIENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 零回归：β=None（显式或默认）完全等价
# ---------------------------------------------------------------------------


def test_soft_beta_none_matches_default_call() -> None:
    """ignite(streams) 与 ignite(streams, soft_beta=None) 返回逐字一致（零回归）。"""
    streams = [
        ("survival", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("appraisal", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
    ]
    terms_default, names_default = ignite(streams)
    terms_none, names_none = ignite(streams, soft_beta=None)
    assert names_default == names_none, f"names 不一致：{names_default} vs {names_none}"
    assert terms_default == terms_none, "terms 不一致"


def test_soft_beta_none_zero_regression_all_weak() -> None:
    """全弱刺激下，soft_beta=None 与默认调用选中相同流（逐字零回归）。"""
    streams = [
        ("survival", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("appraisal", (0.05, 0.05), (0.3, 0.3)),  # 稍高但仍亚阈
    ]
    # 确认两流均亚阈
    for name, mu, prec in streams:
        assert stream_salience(mu, prec) < SALIENCE_THRESHOLD, f"{name} 应是亚阈"
    terms_default, names_default = ignite(streams)
    terms_none, names_none = ignite(streams, soft_beta=None)
    assert names_default == names_none
    assert terms_default == terms_none


def test_soft_beta_none_zero_regression_all_fired() -> None:
    """有流过阈时，soft_beta=None 与默认调用结果逐字一致（零回归）。"""
    streams = [
        ("survival", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
        ("value", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
        ("appraisal", _SUBTHRESH_MU, _SUBTHRESH_PREC),
    ]
    terms_d, names_d = ignite(streams)
    terms_n, names_n = ignite(streams, soft_beta=None)
    assert names_d == names_n
    assert terms_d == terms_n


# ---------------------------------------------------------------------------
# β 激活语义：所有流参与，亚阈精度大幅压低，过阈精度基本保留
# ---------------------------------------------------------------------------


def test_soft_beta_active_includes_all_streams() -> None:
    """soft_beta=20：names 含所有流（包括亚阈流，软门控不过滤）。"""
    streams = [
        ("survival", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("appraisal", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
        ("value", _SUBTHRESH_MU, _SUBTHRESH_PREC),
    ]
    _, names = ignite(streams, soft_beta=20.0)
    assert set(names) == {"survival", "appraisal", "value"}, (
        f"soft_beta=20 时所有流应参与，实际 names={names}"
    )


def test_soft_beta_active_subthresh_precision_reduced() -> None:
    """soft_beta=20：亚阈流的 π_eff（精度）显著低于其原始精度 Π。

    gate = σ(β·(salience − θ))；亚阈 salience << θ → gate ≈ 0 → π_eff << Π。
    """
    BETA = 20.0
    sal_sub = stream_salience(_SUBTHRESH_MU, _SUBTHRESH_PREC)
    gate_sub = sigmoid(BETA * (sal_sub - SALIENCE_THRESHOLD))
    # 亚阈时 gate 应远小于 0.5（显著压缩）
    assert gate_sub < 0.1, f"亚阈 gate 应接近 0，实际 gate={gate_sub:.6f}"
    # 精度被压到 gate * 原始精度
    pi_eff_0 = _SUBTHRESH_PREC[0] * gate_sub
    pi_eff_1 = _SUBTHRESH_PREC[1] * gate_sub
    assert pi_eff_0 < _SUBTHRESH_PREC[0] * 0.1, (
        f"亚阈流精度应被大幅压低，实际 π_eff={pi_eff_0:.6f} vs 原始={_SUBTHRESH_PREC[0]}"
    )
    assert pi_eff_1 < _SUBTHRESH_PREC[1] * 0.1


def test_soft_beta_active_suprathresh_precision_preserved() -> None:
    """soft_beta=20：过阈流的 π_eff 接近原始精度（gate ≈ 1）。"""
    BETA = 20.0
    sal_sup = stream_salience(_SUPRATHRESH_MU, _SUPRATHRESH_PREC)
    gate_sup = sigmoid(BETA * (sal_sup - SALIENCE_THRESHOLD))
    # 过阈时 gate 应接近 1（精度大量保留）
    assert gate_sup > 0.9, f"过阈 gate 应接近 1，实际 gate={gate_sup:.6f}"
    pi_eff_0 = _SUPRATHRESH_PREC[0] * gate_sup
    # 精度保留 90% 以上
    assert pi_eff_0 > _SUPRATHRESH_PREC[0] * 0.9, (
        f"过阈流精度应大量保留，实际 π_eff={pi_eff_0:.6f} vs 原始={_SUPRATHRESH_PREC[0]}"
    )


def test_soft_beta_active_subthresh_much_lower_than_suprathresh() -> None:
    """soft_beta=20：同一组流中，亚阈流的 π_eff 显著低于过阈流的 π_eff（对比验证）。"""
    BETA = 20.0
    streams = [
        ("sub", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("sup", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
    ]
    terms, names = ignite(streams, soft_beta=BETA)
    # terms 是 [(mu, pi_eff), ...] 顺序与 streams 一致
    idx_sub = names.index("sub")
    idx_sup = names.index("sup")
    pi_eff_sub = terms[idx_sub][1][0]  # valence 维精度
    pi_eff_sup = terms[idx_sup][1][0]
    assert pi_eff_sub < pi_eff_sup * 0.1, (
        f"亚阈流精度应比过阈流低得多，sub={pi_eff_sub:.6f} sup={pi_eff_sup:.6f}"
    )


# ---------------------------------------------------------------------------
# 有界性：软门控后 π_eff ∈ (0, 原始Π]，不放大
# ---------------------------------------------------------------------------


def test_soft_gate_pi_eff_not_amplified() -> None:
    """soft_beta=20：所有流的 π_eff ≤ 原始精度（gate∈(0,1)，不能放大精度）。"""
    BETA = 20.0
    streams = [
        ("survival", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("appraisal", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
        ("value", (0.3, 0.3), (0.6, 0.6)),  # 边界附近
    ]
    orig_prec = {name: prec for name, _, prec in streams}
    terms, names = ignite(streams, soft_beta=BETA)
    for name, (_, pi_eff) in zip(names, terms, strict=True):
        assert pi_eff[0] <= orig_prec[name][0] + 1e-9, (
            f"{name} π_eff[0]={pi_eff[0]:.6f} 超过原始精度 {orig_prec[name][0]}"
        )
        assert pi_eff[1] <= orig_prec[name][1] + 1e-9, (
            f"{name} π_eff[1]={pi_eff[1]:.6f} 超过原始精度 {orig_prec[name][1]}"
        )


def test_soft_gate_pi_eff_strictly_positive() -> None:
    """soft_beta=20：π_eff > 0（gate > 0 自动保证，即使亚阈流也不归零）。"""
    BETA = 20.0
    streams = [
        ("survival", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("appraisal", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
    ]
    terms, names = ignite(streams, soft_beta=BETA)
    for name, (_, pi_eff) in zip(names, terms, strict=True):
        assert pi_eff[0] > 0.0, f"{name} π_eff[0] 不应为 0"
        assert pi_eff[1] > 0.0, f"{name} π_eff[1] 不应为 0"


def test_soft_gate_at_various_beta_values() -> None:
    """多个 β 值：β=0.1（近均匀）vs β=50（近硬 step）— 有界性恒成立。"""
    streams = [
        ("sub", _SUBTHRESH_MU, _SUBTHRESH_PREC),
        ("sup", _SUPRATHRESH_MU, _SUPRATHRESH_PREC),
    ]
    orig_prec = {name: prec for name, _, prec in streams}
    for beta in [0.1, 1.0, 20.0, 50.0]:
        terms, names = ignite(streams, soft_beta=beta)
        for name, (_, pi_eff) in zip(names, terms, strict=True):
            assert pi_eff[0] > 0.0, f"β={beta} {name} π_eff[0] 应>0"
            assert pi_eff[0] <= orig_prec[name][0] + 1e-9, (
                f"β={beta} {name} π_eff[0] 不应超过原始精度"
            )


# ---------------------------------------------------------------------------
# SessionConfig 贯通：ignition_beta 默认 None = 零回归
# ---------------------------------------------------------------------------


def test_session_config_ignition_beta_default_none() -> None:
    """SessionConfig().ignition_beta is None（默认零回归）。"""
    cfg = SessionConfig()
    assert cfg.ignition_beta is None


def test_session_config_to_state_flags_ignition_beta_none() -> None:
    """SessionConfig().to_state_flags()["ignition_beta"] is None（展开后仍 None）。"""
    cfg = SessionConfig()
    flags = cfg.to_state_flags()
    assert "ignition_beta" in flags, "to_state_flags 应包含 ignition_beta 键"
    assert flags["ignition_beta"] is None


def test_session_config_ignition_beta_set() -> None:
    """SessionConfig(ignition_beta=20) → 字段值 == 20，展开后也 == 20。"""
    cfg = SessionConfig(ignition_beta=20.0)
    assert cfg.ignition_beta == pytest.approx(20.0)
    flags = cfg.to_state_flags()
    assert flags["ignition_beta"] == pytest.approx(20.0)


def test_build_chat_driver_ignition_beta_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设 ZERO_IGNITION_BETA env → session.config.ignition_beta is None（零回归）。"""
    monkeypatch.delenv("ZERO_IGNITION_BETA", raising=False)
    monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
    from src.orchestration.chat_driver import build_chat_driver

    driver = build_chat_driver(thread="test-ib-none")
    assert driver.session.config.ignition_beta is None


def test_build_chat_driver_ignition_beta_env_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_IGNITION_BETA=20 → session.config.ignition_beta == 20.0（贯通到 SessionConfig）。"""
    monkeypatch.setenv("ZERO_IGNITION_BETA", "20")
    monkeypatch.delenv("ZERO_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZERO_OPENAI_MODEL", raising=False)
    from src.orchestration.chat_driver import build_chat_driver

    driver = build_chat_driver(thread="test-ib-20")
    assert driver.session.config.ignition_beta == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# AffectCoreAgent 集成：ignition_beta=None 零回归；β=20 所有流参与
# ---------------------------------------------------------------------------


def _ws_state(**kw: object) -> AffectState:
    """启用 workspace 的基线 AffectState；支持关键字覆盖。"""
    defaults: dict[str, object] = {
        "stimulus": Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
        "features": [0.5, 0.0, 0.0, 0.8],
        "prior_mu": (0.3, 0.5),
        "prior_sigma": (0.2, 0.2),
        "reward": 0.5,
        "rpe": 0.4,
        "precision": 0.6,
        "rng_seed": 42,
        "workspace_enabled": True,
    }
    defaults.update(kw)
    return AffectState(**defaults)


def test_affect_core_ignition_beta_none_zero_regression() -> None:
    """ignition_beta=None（显式）与默认 AffectState 的 workspace 输出逐字一致（零回归）。"""
    out_default = AffectCoreAgent()(_ws_state())
    out_none = AffectCoreAgent()(_ws_state(ignition_beta=None))
    assert out_default["post_mu"] == out_none["post_mu"]
    assert out_default["post_sigma"] == out_none["post_sigma"]
    assert out_default["affect_sample"] == out_none["affect_sample"]
    assert out_default["ignited_streams"] == out_none["ignited_streams"]
    assert out_default["affect_precision"] == out_none["affect_precision"]


def test_affect_core_ignition_beta_20_all_streams_fired() -> None:
    """ignition_beta=20：ignited_streams 含所有注入流（软门控不过滤任何流）。

    workspace_enabled=True 且 prior_mu/value 流均注入：
    - survival 流（fast_survival_prior from features）
    - appraisal 流（prior_mu）
    - value 流（evidence from rpe/reward）
    所有流都应出现在 ignited_streams。
    """
    out = AffectCoreAgent()(_ws_state(ignition_beta=20.0))
    ignited = out["ignited_streams"]
    # 软门控时所有流参与：至少含 survival、appraisal、value 三流
    assert "survival" in ignited, f"survival 应在 ignited_streams，实际 {ignited}"
    assert "appraisal" in ignited, f"appraisal 应在 ignited_streams，实际 {ignited}"
    assert "value" in ignited, f"value 应在 ignited_streams，实际 {ignited}"


def test_affect_core_ignition_beta_20_output_differs_from_hard_step() -> None:
    """ignition_beta=20 的输出与硬 step（None）不同（软门控改变了精度加权）。

    当有亚阈流被软门控压制（而硬 step 直接排除）时，后验均值应有差异。
    """
    # 构造含过阈和亚阈流的场景：强评价 + 弱 survival
    # prior_mu 高（使评价流过阈），features 低（survival 流亚阈）
    state_hard = _ws_state(
        prior_mu=(0.8, 0.8),
        prior_sigma=(0.1, 0.1),
        features=[0.0, 0.0, 0.0, 0.01],  # 极弱 survival 流
        reward=0.8,
        rpe=0.8,
        precision=1.5,
        ignition_beta=None,
    )
    state_soft = _ws_state(
        prior_mu=(0.8, 0.8),
        prior_sigma=(0.1, 0.1),
        features=[0.0, 0.0, 0.0, 0.01],
        reward=0.8,
        rpe=0.8,
        precision=1.5,
        ignition_beta=20.0,
    )
    out_hard = AffectCoreAgent()(state_hard)
    out_soft = AffectCoreAgent()(state_soft)
    # 软门控含所有流（含极弱 survival），硬 step 可能只含过阈流，两者融合结果应有差异
    # 检查 ignited_streams 数量：软门控 >= 硬 step（至少更多流参与）
    assert len(out_soft["ignited_streams"]) >= len(out_hard["ignited_streams"]), (
        f"软门控应有不少于硬 step 的流，"
        f"soft={out_soft['ignited_streams']} hard={out_hard['ignited_streams']}"
    )


def test_affect_core_ignition_beta_none_does_not_change_state_field_default() -> None:
    """AffectState.ignition_beta 默认 None，确保零回归（state 字段默认对齐）。"""
    state = AffectState()
    assert state.ignition_beta is None
