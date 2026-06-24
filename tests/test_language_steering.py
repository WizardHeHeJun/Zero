"""SteeringLanguageModel 单测：steering 纯数学 + 注入 fake backend 的编排，torch-free。

实机（HF 模型 + hook）路径以 importorskip + 模型探测优雅 skip（本机一般无权重）。
"""

from __future__ import annotations

import os

import pytest

from src.agents.language import LanguageDraft, LanguageModel
from src.agents.language_steering import (
    SteeringLanguageModel,
    axis_from_contrast,
    l2_normalize,
    orthogonalize,
    steering_delta,
)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_axis_from_contrast_is_mean_difference() -> None:
    emo = [[2.0, 0.0], [4.0, 2.0]]  # mean (3,1)
    neu = [[1.0, 1.0], [1.0, 1.0]]  # mean (1,1)
    assert axis_from_contrast(emo, neu) == [2.0, 0.0]


def test_l2_normalize_unit_length_and_zero_safe() -> None:
    v = l2_normalize([3.0, 4.0])
    assert _dot(v, v) == pytest.approx(1.0)
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # 零向量不除零


def test_orthogonalize_removes_reference_component() -> None:
    target = [1.0, 1.0]
    reference = [1.0, 0.0]
    ortho = orthogonalize(target, reference)
    assert _dot(ortho, reference) == pytest.approx(0.0)  # 结果 ⟂ reference


def test_steering_delta_scales_and_is_linear() -> None:
    wv = [1.0, 0.0]
    wa = [0.0, 1.0]
    # 纯 valence（arousal=0）→ delta 沿 w_valence，幅度 = alpha*V
    d = steering_delta(0.5, 0.0, wv, wa, alpha=0.4)
    assert d == [pytest.approx(0.2), pytest.approx(0.0)]
    # 纯 arousal → 沿 w_arousal
    d2 = steering_delta(0.0, 0.5, wv, wa, alpha=0.4)
    assert d2 == [pytest.approx(0.0), pytest.approx(0.2)]
    # alpha 线性缩放
    big = steering_delta(0.5, 0.5, wv, wa, alpha=0.8)
    small = steering_delta(0.5, 0.5, wv, wa, alpha=0.4)
    assert big == [pytest.approx(2 * s) for s in small]


def test_steering_delta_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        steering_delta(0.1, 0.1, [1.0, 0.0], [0.0, 1.0, 0.0])


class _FakeBackend:
    """捕获传入的 steering delta，返回含某情绪词的固定文本（供反推情感断言）。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_delta: list[float] | None = None

    def generate_steered(self, prompt: str, delta: list[float]) -> str:
        self.last_delta = delta
        return self.text


async def test_model_applies_steering_and_appraises_back() -> None:
    backend = _FakeBackend(text="我感到非常愤怒")
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0], w_arousal=[0.0, 1.0], alpha=0.5, backend=backend
    )
    draft = await model.generate(affect=(-0.8, 0.7), context="冲突", retrieved="", feedback=None)
    assert isinstance(draft, LanguageDraft)
    # 反推情感来自词典（文本含「愤怒」(-0.8,0.7)）→ 负效价高唤起
    assert draft.affect[0] < 0.0 and draft.affect[1] > 0.0
    # backend 收到的 delta = α(V·w_V + A·w_A)
    assert backend.last_delta == steering_delta(-0.8, 0.7, [1.0, 0.0], [0.0, 1.0], alpha=0.5)


async def test_model_orthogonalizes_arousal_axis() -> None:
    # 传入非正交两轴；构造后 arousal 轴应被正交化为 ⟂ valence 轴
    model = SteeringLanguageModel(
        w_valence=[1.0, 0.0], w_arousal=[1.0, 1.0], backend=_FakeBackend("平静")
    )
    assert _dot(model.w_valence, model.w_arousal) == pytest.approx(0.0)


def test_satisfies_language_model_protocol() -> None:
    model: LanguageModel = SteeringLanguageModel(
        w_valence=[1.0, 0.0], w_arousal=[0.0, 1.0], backend=_FakeBackend("x")
    )
    assert hasattr(model, "generate")


def test_transformers_backend_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model_name = os.getenv("ZERO_STEER_MODEL")
    if not model_name:
        pytest.skip("未设 ZERO_STEER_MODEL（HF 因果 LM）→ 跳过实机 steering smoke")
    from src.agents.language_steering import _TransformersSteerBackend

    backend = _TransformersSteerBackend(model_name, target_layer=-1)
    out = backend.generate_steered("你好", [0.0])  # 零 delta：仅验证 hook 装卸与生成不崩
    assert isinstance(out, str)
