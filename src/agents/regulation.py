"""RegulationAgent：lPFC 调控节点（掩饰/再评价）。

对应外侧前额叶。默认恒等（自发=随意）；开启后按社会显示规则压制效价强度、
降低唤醒，使「随意通路」偏离「自发通路」——这是真笑/假笑差异的来源。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import clamp
from src.orchestration.state import AffectState


class RegulationAgent:
    """对采样情绪施加掩饰/再评价策略，产出随意通路所用的 regulated_affect。"""

    def __call__(self, state: AffectState) -> dict:
        if state.affect_sample is None:
            return {}
        valence, arousal = state.affect_sample
        if state.regulation_enabled:
            regulated = (clamp(0.3 * valence, -1.0, 1.0), clamp(0.5 * arousal, -1.0, 1.0))
        else:
            regulated = (valence, arousal)
        entry = {
            "node": "regulation",
            "regulated_affect": regulated,
            "enabled": state.regulation_enabled,
        }
        return {"regulated_affect": regulated, "trace": [entry]}
