"""zero-link Q1 拍板（2026-07-14）：prosody 量纲标记 `prosody_scale` 契约测试。

背景：`prosody` 通道量纲双方言——解析占位（affect_math.decode_channels）与整向量通路
（expression_decoder.vector_to_channels）出**倍率口径** "ratio"（speech_rate/pitch 以 1.0
为基线、energy∈[0,1]）；专用 ProsodyDecoder（sigmoid）出**归一 [0,1]** "normalized"。

拍板：canonical 目标口径=normalized；过渡期各发射点诚实标注，MCP 情感 TTS mapper 按
`expression["prosody_scale"]` 分支消费、收窄校验。标记作 channels **兄弟键**（不塞进 prosody
子 dict），保 prosody 通道纯 3 值——占位路径 torch-free。
"""

from __future__ import annotations

from typing import Any

from src.agents.affect_math import decode_channels
from src.agents.expression import ExpressionAgent
from src.agents.models.composite import CompositeChannelDecoder
from src.orchestration.state import AffectState


class _FakeProsodyModel:
    """归一 [0,1] 韵律真模型替身（duck-typed·torch-free）——触发 composite 的 normalized 分支。"""

    def predict_prosody(self, valence: float, arousal: float) -> dict[str, float]:
        return {"speech_rate": 0.5, "pitch": 0.5, "energy": 0.5}


class _UntaggedDecoder:
    """未标注量纲的注入 decoder（mock 先例）——验证 hoist 保持 additive 零回归。"""

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        return {"facs_au": {}, "text_label": "x", "physiology": {}, "prosody": {}}


def test_analytic_placeholder_tagged_ratio() -> None:
    # 解析占位（默认路径）出倍率口径 → "ratio"
    assert decode_channels((0.5, 0.3))["prosody_scale"] == "ratio"


def test_analytic_tag_present_under_both_facs_gates() -> None:
    # prosody 通道与 facs_extended 门控无关 → 两门控下均带 tag、均 "ratio"（零回归）
    assert decode_channels((0.5, 0.3), facs_extended=False)["prosody_scale"] == "ratio"
    assert decode_channels((-0.4, 0.6), facs_extended=True)["prosody_scale"] == "ratio"


def test_analytic_prosody_subdict_stays_pure_three_values() -> None:
    # 标记是兄弟键——prosody 子 dict 仍纯 3 值（不污染 MCP ProsodyChannel 的数值校验）
    assert set(decode_channels((0.5, 0.3))["prosody"]) == {"speech_rate", "pitch", "energy"}


def test_composite_fallback_ratio_and_real_normalized() -> None:
    # 未注入韵律真模型 → 回退解析占位 → "ratio"
    assert CompositeChannelDecoder().predict_channels(0.5, 0.3)["prosody_scale"] == "ratio"
    # 注入专用韵律真模型 → 覆盖为 "normalized"（canonical 目标口径）
    real = CompositeChannelDecoder(prosody_model=_FakeProsodyModel())
    assert real.predict_channels(0.5, 0.3)["prosody_scale"] == "normalized"


def test_expression_hoists_prosody_scale_to_top_level() -> None:
    # 两头共用同一 decoder → 同量纲；顶层 hoist 供 MCP 单点读，两头也各自自描述
    agent = ExpressionAgent(decoder=CompositeChannelDecoder(prosody_model=_FakeProsodyModel()))
    expr = agent(AffectState(affect_sample=(0.6, 0.4)))["expression"]
    assert expr["prosody_scale"] == "normalized"
    assert expr["spontaneous"]["prosody_scale"] == "normalized"
    assert expr["voluntary"]["prosody_scale"] == "normalized"


def test_expression_default_decoder_tagged_ratio() -> None:
    # 未注入 decoder → ExpressionAgent 走 decode_channels 占位 → 顶层 "ratio"
    expr = ExpressionAgent()(AffectState(affect_sample=(0.6, 0.4)))["expression"]
    assert expr["prosody_scale"] == "ratio"


def test_untagged_decoder_hoist_is_additive_noop() -> None:
    # 注入未标注量纲的 decoder（mock）→ 顶层不挂 prosody_scale（additive 零回归）
    expr = ExpressionAgent(decoder=_UntaggedDecoder())(AffectState(affect_sample=(0.6, 0.4)))[
        "expression"
    ]
    assert "prosody_scale" not in expr


def test_both_heads_carry_consistent_scale_default_path() -> None:
    # W1 硬化（code-reviewer 2026-07-14）：两头共用同一无状态 decoder → 量纲不变式一致。
    # 顶层 hoist 从 spontaneous 单点读，此断言锁定 voluntary 与 spontaneous 量纲一致，
    # 防将来 decoder 引入有状态路径时两头静默分叉（默认占位路径同样覆盖）。
    expr = ExpressionAgent()(AffectState(affect_sample=(0.6, 0.4)))["expression"]
    assert expr["spontaneous"]["prosody_scale"] == "ratio"
    assert expr["voluntary"]["prosody_scale"] == "ratio"
    assert expr["voluntary"]["prosody_scale"] == expr["spontaneous"]["prosody_scale"]
