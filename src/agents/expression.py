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

    def _decode(self, affect: tuple[float, float]) -> dict[str, Any]:
        if self.decoder is not None:
            return self.decoder.predict_channels(affect[0], affect[1])
        return decode_channels(affect)

    def __call__(self, state: AffectState) -> dict:
        if state.affect_sample is None:
            return {}
        spontaneous = self._decode(state.affect_sample)
        voluntary_source = (
            state.regulated_affect if state.regulated_affect is not None else state.affect_sample
        )
        voluntary = self._decode(voluntary_source)
        expression: dict[str, Any] = {
            "valence_arousal": state.affect_sample,
            "spontaneous": spontaneous,  # 非随意通路（直连 AffectCore）
            "voluntary": voluntary,  # 随意通路（经 Regulation）
        }
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
