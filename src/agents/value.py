"""ValueAgent：在线 TD 价值节点（价值/精度辅佐）。

对应 VTA/腹侧纹状体多巴胺系统。维护 V(s) 并在线更新（骨架里唯一真学习组件），
产出 RPE δ 与精度 π。V(s) 是**运行态**，随 AffectState 进 Checkpointer，
不写入长期记忆图谱。节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.agents.affect_math import precision, td_update
from src.orchestration.state import AffectState


class ValueAgent:
    """以 stimulus 名为情境键，在线 TD 更新 V(s)，产出 δ 与精度 π。"""

    def __call__(self, state: AffectState) -> dict:
        if state.reward is None or state.stimulus is None:
            return {}
        key = state.stimulus.name
        delta, new_value, table = td_update(state.value_table, key, state.reward)
        # precision_commensurable（议会 2026-07-28 第四轮；默认关=逐字旧行为）：门开时把
        # sigmoid 概率重标定成逆方差量纲。此处产出的 pi 是**默认融合路径**
        # （affect_core.py 的 gaussian_fuse 分支）的证据精度，每轮无条件生效。
        pi = precision(delta, new_value, commensurable=state.precision_commensurable)
        entry = {
            "node": "value",
            "rpe": delta,
            "value_estimate": new_value,
            "precision": pi,
        }
        return {
            "rpe": delta,
            "value_estimate": new_value,
            "value_table": table,
            "precision": pi,
            "trace": [entry],
        }
