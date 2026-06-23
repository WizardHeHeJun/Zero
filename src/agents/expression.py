"""ExpressionAgent：面神经双通路 + 自主神经输出（多通道）。

对应面神经双通路与自主神经系统。
- 自发头（非随意通路）：直接由 AffectCore 的 e* 解码。
- 随意头（随意通路）：由 RegulationAgent 的 regulated_affect 解码。
二者不一致即「真笑/假笑」。每头各产出 4 通道（FACS AU/文本标签/生理/韵律）。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import decode_channels
from src.orchestration.state import AffectState


class ExpressionAgent:
    """双头解码：自发头(e*) 与 随意头(regulated) × 4 通道。"""

    def __call__(self, state: AffectState) -> dict:
        if state.affect_sample is None:
            return {}
        spontaneous = decode_channels(state.affect_sample)
        voluntary_source = (
            state.regulated_affect if state.regulated_affect is not None else state.affect_sample
        )
        voluntary = decode_channels(voluntary_source)
        expression = {
            "valence_arousal": state.affect_sample,
            "spontaneous": spontaneous,  # 非随意通路（直连 AffectCore）
            "voluntary": voluntary,  # 随意通路（经 Regulation）
        }
        entry = {"node": "expression", "valence_arousal": state.affect_sample}
        return {"expression": expression, "trace": [entry]}
