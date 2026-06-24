"""MoodAgent：慢变心境更新节点（A.7 时间深度/滞后）。

承接 AffectCore 采样出的 e*，按双稳动力学 `mood_step` 更新慢变心境 `mood`——
持续负向把心境推入负盆且难以拉出（历史依赖/滞后）。`mood` 是运行态，随
Checkpointer 跨刺激持久化、回灌下一轮 AffectCore 的融合；**不入图谱**。
`mood_enabled` 关闭或无 e* 时为 no-op。节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import mood_step
from src.orchestration.state import AffectState


class MoodAgent:
    """用 e* 驱动心境的双稳更新；mood 跨刺激持久化形成滞后。"""

    def __call__(self, state: AffectState) -> dict:
        if not state.mood_enabled or state.affect_sample is None:
            return {}
        prev_mood = state.mood if state.mood is not None else (0.0, 0.0)
        new_mood = mood_step(prev_mood, state.affect_sample)
        entry = {"node": "mood", "mood": new_mood}
        return {"mood": new_mood, "trace": [entry]}
