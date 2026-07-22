"""RegulationAgent：lPFC 调控节点（表达抑制 / 认知重评）。

对应外侧前额叶。默认恒等（自发=随意）；开启后按所选策略改写「随意通路」，使其偏离
「自发通路」——真笑/假笑差异的来源。两种策略对应 Gross 过程模型的两个干预点
（Gross 1998 DOI:10.1037/0022-3514.74.1.224；Gross & John 2003）：
- suppression（默认，晚期）：反应调制（response modulation）末端干预——仅按比例压制效价/唤醒
  幅度作用于**表达侧 regulated_affect**（随意通路输出·语言层消费），**不改内核 affect_sample
  体验本身**（Gross 抑制=压末端表达不改内在体验·此处 regulated_affect 承表达侧角色故忠实）。
- reappraisal（早期）：前因聚焦重解释情境意义，负效价上抬、唤醒平复（改体验，更省力更有效）。
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
