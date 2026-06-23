"""PerceptionAgent：感知节点，由 stimulus 提取特征向量。

对应感觉皮层 + 杏仁核早期。节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.orchestration.state import AffectState


class PerceptionAgent:
    """从 stimulus 抽取特征（第一版直接取 OCC 评价维度作为占位特征）。"""

    def __call__(self, state: AffectState) -> dict:
        stim = state.stimulus
        if stim is None:
            return {}
        features = [
            stim.goal_congruence,
            stim.standard_compliance,
            stim.attitude_appeal,
            stim.intensity,
        ]
        entry = {"node": "perception", "features": features}
        return {"features": features, "trace": [entry]}
