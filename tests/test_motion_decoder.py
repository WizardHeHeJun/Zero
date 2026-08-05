"""MotionDecoder 与运行时守卫：锚点包络之外不得饱和/失控。

数学席要求的守卫（仿 `test_facs_runtime_guards.py`）——训练集只覆盖 15 个离散锚点，
运行时输入是连续 (v,a)，**包络外的行为不受数据约束**，必须由测试兜住。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.agents.models.motion_decoder import AnchorInterpolator, MotionMLP  # noqa: E402

_ANCHORS = torch.tensor(
    [[0.0, 0.0], [0.7, 0.5], [-0.6, 0.7], [-0.6, -0.4], [0.3, -0.5]], dtype=torch.float32
)
_K = 3


def _grid(step: float = 0.25) -> torch.Tensor:
    """[-1,1]² 全网格：覆盖锚点包络内外。"""
    axis = torch.arange(-1.0, 1.0 + 1e-6, step)
    vv, aa = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([vv.flatten(), aa.flatten()], dim=1)


def test_interpolator_rejects_bad_anchor_shape() -> None:
    with pytest.raises(ValueError):
        AnchorInterpolator(torch.zeros(5), out_dim=_K)


def test_output_shape_and_positivity() -> None:
    """调制系数是**乘性缩放**，必须恒正——为负会让动作反向。"""
    model = AnchorInterpolator(_ANCHORS, out_dim=_K)
    out = model(_grid())
    assert out.shape == (_grid().shape[0], _K)
    assert bool((out > 0).all()), "调制系数出现非正值"


def test_no_saturation_outside_anchor_envelope() -> None:
    """⚠ 运行时守卫：包络外输出不得爆炸或塌成常数。

    训练只见过 15 个锚点，运行时 (v,a) 连续 —— 包络外没有数据约束，
    只能靠结构（RBF softmax 权重和为 1）保证输出落在锚点值的凸包内。
    """
    model = AnchorInterpolator(_ANCHORS, out_dim=_K)
    with torch.no_grad():
        model.values.copy_(torch.randn(_ANCHORS.shape[0], _K))
    out = model(_grid(step=0.1))
    assert bool(torch.isfinite(out).all())
    # softplus(凸组合) 的上界由锚点值决定，不会随距离发散
    ceiling = torch.nn.functional.softplus(model.values.max()) * 1.05
    assert bool((out <= ceiling).all()), "包络外出现超过锚点上界的输出（失控）"


def test_output_varies_across_input() -> None:
    """不得退化成常数——那等于「模型没在工作」却仍然绿。"""
    model = AnchorInterpolator(_ANCHORS, out_dim=_K)
    with torch.no_grad():
        model.values.copy_(torch.tensor([[3.0, 0.0, 0.0]] * 1 + [[-3.0, 0.0, 0.0]] * 4))
    with torch.no_grad():
        out = model(_ANCHORS)
    assert float(out[:, 0].std()) > 1e-3


def test_prediction_near_anchor_follows_that_anchor() -> None:
    """靠近某锚点时预测应偏向该锚点的值（插值语义成立的最低要求）。"""
    model = AnchorInterpolator(_ANCHORS, out_dim=1)
    with torch.no_grad():
        model.values.copy_(torch.tensor([[5.0], [-5.0], [-5.0], [-5.0], [-5.0]]))
        model.log_bandwidth.copy_(torch.tensor(-1.5))  # 窄带宽 → 更接近最近邻
    near_first = model(torch.tensor([[0.02, 0.02]], dtype=torch.float32))
    near_second = model(torch.tensor([[0.68, 0.48]], dtype=torch.float32))
    assert float(near_first[0, 0]) > float(near_second[0, 0])


def test_parameter_count_matches_anchor_count() -> None:
    """自由度 = 锚点数 × K + 1（带宽）——数学席对结构性不可辨识的处方，钉死防漂移。

    若有人把它改成大 MLP，这条会红。
    """
    model = AnchorInterpolator(_ANCHORS, out_dim=_K)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == _ANCHORS.shape[0] * _K + 1


def test_mlp_baseline_runs() -> None:
    """对照组可跑（消融要用），输出同样恒正。"""
    out = MotionMLP(out_dim=_K)(_grid())
    assert out.shape[1] == _K
    assert bool((out > 0).all())
