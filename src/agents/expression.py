"""ExpressionAgent：面神经双通路 + 自主神经输出（多通道）。

对应面神经双通路与自主神经系统。
- 自发头（非随意通路）：直接由 AffectCore 的 e* 解码。
- 随意头（随意通路）：由 RegulationAgent 的 regulated_affect 解码。
二者不一致即「真笑/假笑」。每头各产出 4 通道（FACS AU/文本标签/生理/韵律）。

解码器可注入：注入训练好的 `ChannelDecoder`（如 torch 版 ExpressionDecoder）走真网络，
未注入则回退到 `affect_math.decode_channels` 解析占位。本模块保持 torch-free。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from typing import Any, Protocol

from src.agents.affect_math import decode_channels
from src.orchestration.state import AffectState


class ChannelDecoder(Protocol):
    """通道解码器协议（鸭子类型）；torch 版 ExpressionDecoder 结构上满足。"""

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]: ...


class ExpressionAgent:
    """双头解码：自发头(e*) 与 随意头(regulated) × 4 通道。"""

    def __init__(self, decoder: ChannelDecoder | None = None) -> None:
        self.decoder = decoder

    def _decode(
        self,
        affect: tuple[float, float],
        *,
        coping_potential: float = 0.0,
        facs_extended: bool = False,
    ) -> dict[str, Any]:
        if self.decoder is not None:
            # 议会遗留 2 设计门（方案 b）：若注入 decoder 有可选方法 predict_channels_coping，
            # 传入当轮 coping/facs_extended（真 facs_model 注入后 C2 residual 才拿到正确 coping）；
            # 否则回退 predict_channels(v,a)（旧 decoder/mock 零改动）。
            # additive 非 breaking：ChannelDecoder 协议不变（CS 席约束 #4）。
            # 探测得到的方法假定签名 (v, a, coping, facs_extended) -> dict（本项目
            # CompositeChannelDecoder.predict_channels_coping 即此签名）；getattr 返回 Any，
            # 静态检查放行，调用方须保证注入 decoder 的该方法签名一致。
            coping_aware = getattr(self.decoder, "predict_channels_coping", None)
            if coping_aware is not None:
                result: dict[str, Any] = coping_aware(
                    affect[0], affect[1], coping_potential, facs_extended
                )
                return result
            return self.decoder.predict_channels(affect[0], affect[1])
        return decode_channels(
            affect, coping_potential=coping_potential, facs_extended=facs_extended
        )

    def __call__(self, state: AffectState) -> dict:
        if state.affect_sample is None:
            return {}
        # 占位路径：透传 coping_potential_state + facs_extended（防空悬，W1）
        coping = state.coping_potential_state
        facs_ext = state.facs_extended
        # 自发头（push·锥体外路·皮层下驱动）：全量传 coping，coping-driven AU 原始强度泄漏。
        spontaneous = self._decode(
            state.affect_sample, coping_potential=coping, facs_extended=facs_ext
        )
        voluntary_source = (
            state.regulated_affect if state.regulated_affect is not None else state.affect_sample
        )
        # 随意头（pull·锥体束意志调控）差异化（议会 C1 设计门 2026-07-14）：coping-driven AU
        # 的泄漏按 voluntary_coping_leak∈[0,1] 衰减（意志可部分压制但不归零，Rinn 1984）。
        # 默认 leak=1.0 → voluntary_coping==coping → 两头等值 = 逐字旧行为（零回归）。
        voluntary_coping = coping * state.voluntary_coping_leak
        voluntary = self._decode(
            voluntary_source, coping_potential=voluntary_coping, facs_extended=facs_ext
        )
        expression: dict[str, Any] = {
            "valence_arousal": state.affect_sample,
            "spontaneous": spontaneous,  # 非随意通路（直连 AffectCore）
            "voluntary": voluntary,  # 随意通路（经 Regulation）
        }
        # prosody 量纲标记 hoist 到 expression 顶层（zero-link Q1 拍板 2026-07-14）：
        # 两头（spontaneous/voluntary）共用同一**无状态** decoder → 量纲一致，
        # 故从 spontaneous 单点读即代表两头。⚠ 若 decoder 将来引入 per-call 有状态
        # 路径（dropout/温度采样）致两头量纲分叉，须重审（见 test_prosody_scale_tag
        # 两头不变式断言）。供 MCP mapper 单点读 expression["prosody_scale"]；未标注
        # 量纲的注入 decoder（如 mock）不挂键 = additive 零回归。
        if "prosody_scale" in spontaneous:
            expression["prosody_scale"] = spontaneous["prosody_scale"]
        # 语言层开启时，把生成的语言内容并入最终表现（情感↔语言相互判断的产物）
        if state.language_text is not None:
            expression["language"] = {
                "text": state.language_text,
                "affect": state.language_affect,
                "iters": state.language_iter,
                "consistency": state.language_consistency,
            }
        entry = {"node": "expression", "valence_arousal": state.affect_sample}
        return {"expression": expression, "trace": [entry]}
