"""RegulationAgent：lPFC 调控节点（表达抑制 / 认知重评）。

对应外侧前额叶。默认恒等（自发=随意）；开启后按所选策略改写「随意通路」，使其偏离
「自发通路」——真笑/假笑差异的来源。两种策略对应 Gross 过程模型的两个干预点：
- suppression（默认，晚期）：仅按比例压制效价/唤醒幅度（压表达、不改体验）。
- reappraisal（早期）：重新解释情境意义，负效价上抬、唤醒平复（改体验，更省力更有效）。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import clamp, reappraise
from src.orchestration.state import AffectState


class RegulationAgent:
    """对采样情绪施加抑制/重评策略，产出随意通路所用的 regulated_affect。"""

    def __call__(self, state: AffectState) -> dict:
        if state.affect_sample is None:
            return {}
        valence, arousal = state.affect_sample
        if not state.regulation_enabled:
            regulated = (valence, arousal)
        elif state.regulation_strategy == "reappraisal":
            regulated = reappraise((valence, arousal))
        else:  # suppression（默认）
            regulated = (clamp(0.3 * valence, -1.0, 1.0), clamp(0.5 * arousal, -1.0, 1.0))
        entry = {
            "node": "regulation",
            "regulated_affect": regulated,
            "enabled": state.regulation_enabled,
            "strategy": state.regulation_strategy if state.regulation_enabled else "identity",
        }
        return {"regulated_affect": regulated, "trace": [entry]}
