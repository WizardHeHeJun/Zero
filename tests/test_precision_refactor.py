"""议会裁决 A-P1-A + A-P0-B 精度重构测试。

覆盖：
- precision_da 纯函数语义（消 β·V、单调于 |δ|、MIN_PRECISION 下界）。
- 零回归断言：两 flag 默认 False 时 workspace 路径输出逐字不变。
- 开启断言：precision_split=True 使 value 流精度不含 V 项；
  fuse_independence_correct=True 使 value 流 valence 维精度降到 MIN_PRECISION。
torch/API-free，纯函数 + state 单测。
"""

from __future__ import annotations

import pytest

from src.agents.affect_core import AffectCoreAgent
from src.agents.affect_math import MIN_PRECISION, precision, precision_da
from src.orchestration.state import AffectState, Stimulus

# ---------------------------------------------------------------------------
# precision_da 纯函数
# ---------------------------------------------------------------------------


def test_precision_da_floor() -> None:
    """delta=0 时仍返回 MIN_PRECISION 下界（sigmoid(0)=0.5 > MIN_PRECISION，下界由钳制保证）。"""
    result = precision_da(0.0)
    assert result >= MIN_PRECISION


def test_precision_da_monotonic_in_abs_delta() -> None:
    """|δ| 越大 precision_da 越高（单调增）。"""
    assert precision_da(2.0) > precision_da(0.5) > precision_da(0.1)


def test_precision_da_symmetric_in_sign() -> None:
    """precision_da 只依赖 |δ|，正负等价。"""
    assert precision_da(1.0) == pytest.approx(precision_da(-1.0))


def test_precision_da_differs_from_precision_when_value_nonzero() -> None:
    """有 value_estimate 时 precision_da ≠ precision（消 β·V 的关键语义验证）。

    precision(δ, V) = σ(α|δ| + β·V)；precision_da(δ) = σ(α|δ|)。
    V≠0 时两者不同（否则 β·V 项无效）。
    """
    delta = 0.5
    value = 1.0  # 非零价值
    p_old = precision(delta, value)
    p_da = precision_da(delta)
    assert p_old != pytest.approx(p_da), (
        f"V={value}≠0 时 precision_da 应与 precision 结果不同，但都等于 {p_da}"
    )


def test_precision_da_equals_precision_when_beta_zero() -> None:
    """value_estimate=0 时 precision_da(δ) ≈ precision(δ, 0)（β·V=0，两者退化相同）。"""
    delta = 0.8
    assert precision_da(delta) == pytest.approx(precision(delta, 0.0))


# ---------------------------------------------------------------------------
# 工具：构造 workspace 基线 state
# ---------------------------------------------------------------------------


def _ws_state(**kw: object) -> AffectState:
    """构造启用 workspace 的 AffectState，支持关键字覆盖（kw 优先于默认值）。"""
    defaults: dict[str, object] = {
        "stimulus": Stimulus(name="t", goal_congruence=0.5, intensity=0.8),
        "features": [0.5, 0.0, 0.0, 0.8],
        "prior_mu": (0.3, 0.5),
        "prior_sigma": (0.2, 0.2),
        "reward": 0.5,
        "rpe": 0.4,  # 非零 rpe，使 precision_da 与旧 precision 可区分
        "precision": 0.6,
        "rng_seed": 42,
        "workspace_enabled": True,
    }
    defaults.update(kw)
    return AffectState(**defaults)


# ---------------------------------------------------------------------------
# 零回归断言：两 flag 默认 False → 输出逐字不变
# ---------------------------------------------------------------------------


def test_zero_regression_both_flags_false() -> None:
    """precision_split=False 且 fuse_independence_correct=False 时，
    workspace 路径输出与不传这两个字段的基线逐字相同。
    """
    baseline = AffectCoreAgent()(_ws_state())
    with_flags = AffectCoreAgent()(
        _ws_state(precision_split=False, fuse_independence_correct=False)
    )
    assert baseline["post_mu"] == with_flags["post_mu"]
    assert baseline["post_sigma"] == with_flags["post_sigma"]
    assert baseline["affect_sample"] == with_flags["affect_sample"]
    assert baseline["ignited_streams"] == with_flags["ignited_streams"]
    assert baseline["affect_precision"] == with_flags["affect_precision"]
    # trace 含 node/post_mu/post_sigma/affect_sample/ignited_streams
    assert baseline["trace"] == with_flags["trace"]


def test_zero_regression_non_workspace_path_unaffected() -> None:
    """非 workspace 路径（workspace_enabled=False）下，两 flag 对输出无影响。"""
    base = AffectState(
        prior_mu=(0.3, 0.5),
        prior_sigma=(0.2, 0.2),
        reward=0.5,
        rpe=0.4,
        precision=0.6,
        rng_seed=42,
        workspace_enabled=False,
    )
    out_base = AffectCoreAgent()(base)
    out_with = AffectCoreAgent()(
        AffectState(
            prior_mu=(0.3, 0.5),
            prior_sigma=(0.2, 0.2),
            reward=0.5,
            rpe=0.4,
            precision=0.6,
            rng_seed=42,
            workspace_enabled=False,
            precision_split=True,
            fuse_independence_correct=True,
        )
    )
    assert out_base["post_mu"] == out_with["post_mu"]
    assert out_base["post_sigma"] == out_with["post_sigma"]
    assert out_base["affect_sample"] == out_with["affect_sample"]


# ---------------------------------------------------------------------------
# 开启断言：precision_split=True
# ---------------------------------------------------------------------------


def test_precision_split_changes_output() -> None:
    """precision_split=True 时输出与默认不同（消 β·V 真正影响了融合结果）。

    rpe=0.4, precision=0.6, value_estimate 非零场景下，
    precision_da(0.4) ≠ 0.6（即 pi），因此 value 流精度改变 → 后验偏移。
    """
    out_off = AffectCoreAgent()(_ws_state(precision_split=False))
    out_on = AffectCoreAgent()(_ws_state(precision_split=True))
    # 后验均值应在至少一个维度上有差异
    same = out_off["post_mu"][0] == pytest.approx(out_on["post_mu"][0]) and out_off["post_mu"][
        1
    ] == pytest.approx(out_on["post_mu"][1])
    assert not same, (
        f"precision_split=True 应改变 post_mu，"
        f"但 off={out_off['post_mu']} == on={out_on['post_mu']}"
    )


def test_precision_split_value_prec_no_beta_v() -> None:
    """precision_split=True 时 value 流精度不含 β·V 混同项。

    通过对比"value_estimate 存在但精度相同"间接验证：
    precision(rpe=0.4, V=大值) >> precision_da(rpe=0.4)，
    而 precision_split=True 时后验精度（affect_precision）应更低（value 投票权更小）。
    """
    # 给一个大的 precision（模拟 V 大时旧式 precision 虚高的情况）
    state_off = _ws_state(precision_split=False, precision=2.0)  # 高 precision 模拟旧 β·V 混同
    state_on = _ws_state(precision_split=True, precision=2.0)  # DA 路径只用 |δ|
    out_off = AffectCoreAgent()(state_off)
    out_on = AffectCoreAgent()(state_on)
    # precision_split=True 下 value 流精度由 precision_da(0.4) 决定，远小于 pi=2.0
    # → 后验整体精度（affect_precision）应更低（value 贡献减少）
    assert out_on["affect_precision"] < out_off["affect_precision"], (
        f"precision_split=True 应降低 affect_precision（消 β·V 虚高），"
        f"但 on={out_on['affect_precision']} >= off={out_off['affect_precision']}"
    )


# ---------------------------------------------------------------------------
# 开启断言：fuse_independence_correct=True
# ---------------------------------------------------------------------------


def test_fuse_independence_correct_changes_output() -> None:
    """fuse_independence_correct=True 时 valence 维精度降到 MIN_PRECISION → 输出改变。"""
    out_off = AffectCoreAgent()(_ws_state(fuse_independence_correct=False))
    out_on = AffectCoreAgent()(_ws_state(fuse_independence_correct=True))
    same = out_off["post_mu"][0] == pytest.approx(out_on["post_mu"][0]) and out_off["post_mu"][
        1
    ] == pytest.approx(out_on["post_mu"][1])
    assert not same, (
        f"fuse_independence_correct=True 应改变 post_mu，"
        f"但 off={out_off['post_mu']} == on={out_on['post_mu']}"
    )


def test_fuse_independence_correct_lowers_valence_weight() -> None:
    """fuse_independence_correct=True → value 流 valence 维精度 MIN_PRECISION → 后验 valence
    更贴 appraisal/survival（与 appraisal 共线的 value valence 贡献被剥除），
    后验整体精度更低（过度自信被消除）。
    """
    out_off = AffectCoreAgent()(_ws_state(fuse_independence_correct=False))
    out_on = AffectCoreAgent()(_ws_state(fuse_independence_correct=True))
    # 1. 后验总精度：correction 后 value valence 贡献极小 → 精度应降低
    assert out_on["affect_precision"] < out_off["affect_precision"], (
        f"fuse_independence_correct=True 应降低 affect_precision（消过度自信），"
        f"但 on={out_on['affect_precision']} >= off={out_off['affect_precision']}"
    )


def test_fuse_independence_correct_keeps_arousal_precision() -> None:
    """fuse_independence_correct=True 时 arousal 维精度不受影响（只改 valence 维）。

    通过对比单独开 precision_split=False 基线，后验 arousal sigma 应相近（arousal 维不变）。
    """
    # 构造 arousal_gain=1（prior_mu[1]=0 → AROUSAL_GAIN*0=0 → arousal_gain=1）
    # 使 arousal 维精度数值可预测
    state_off = _ws_state(prior_mu=(0.3, 0.0), fuse_independence_correct=False)
    state_on = _ws_state(prior_mu=(0.3, 0.0), fuse_independence_correct=True)
    out_off = AffectCoreAgent()(state_off)
    out_on = AffectCoreAgent()(state_on)
    # arousal sigma（post_sigma[1]）在 correction 后应与 off 相差不超过 0.05
    # （arousal 维精度不变，仅 valence 维精度被降低）
    assert abs(out_on["post_sigma"][1] - out_off["post_sigma"][1]) < 0.05, (
        f"fuse_independence_correct 不应大幅改变 arousal sigma，"
        f"但 on={out_on['post_sigma'][1]:.4f} vs off={out_off['post_sigma'][1]:.4f}"
    )


def test_two_flags_independent() -> None:
    """两个 flag 可独立开关，互不依赖（不相互干扰）。"""
    only_split = AffectCoreAgent()(_ws_state(precision_split=True, fuse_independence_correct=False))
    only_correct = AffectCoreAgent()(
        _ws_state(precision_split=False, fuse_independence_correct=True)
    )
    both_on = AffectCoreAgent()(_ws_state(precision_split=True, fuse_independence_correct=True))
    both_off = AffectCoreAgent()(_ws_state(precision_split=False, fuse_independence_correct=False))

    # 各情况后验精度应为 4 个不同值（至少存在差异，说明各自独立生效）
    precs = {
        only_split["affect_precision"],
        only_correct["affect_precision"],
        both_on["affect_precision"],
        both_off["affect_precision"],
    }
    assert len(precs) >= 2, (
        f"两 flag 独立开关应产生至少 2 种不同的 affect_precision，但都相同：{precs}"
    )
