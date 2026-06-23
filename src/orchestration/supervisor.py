"""SupervisorAgent：协调与任务完成节点。

只做协调 + 任务完成判定，**不含业务逻辑**（业务在各 Worker）。记忆写入
**只在此处**发生（节流，见 memory-rules.md #1）：当前情绪事件写 session 作用域，
长期情绪倾向写 user 作用域，均显式 scope。注入 MemoryClient，不直连图谱。
"""

from __future__ import annotations

from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.state import AffectState


class SupervisorAgent:
    """任务完成节点：标记完成并节流 flush 记忆。"""

    def __init__(self, memory: MemoryClient) -> None:
        self.memory = memory

    async def __call__(self, state: AffectState) -> dict:
        affect = state.affect_sample
        if affect is not None:
            stim_name = state.stimulus.name if state.stimulus is not None else "unknown"
            # 当前情绪事件：session 作用域
            await self.memory.write(
                f"event={stim_name} affect=({affect[0]:.2f},{affect[1]:.2f})",
                scope=Scope.SESSION,
                key=state.session_id,
            )
            # 长期情绪倾向：user 作用域
            value = state.value_estimate if state.value_estimate is not None else 0.0
            await self.memory.write(
                f"disposition stimulus={stim_name} value={value:.3f}",
                scope=Scope.USER,
                key=state.user_id,
            )
        entry = {"node": "supervisor", "task_complete": True}
        return {"task_complete": True, "trace": [entry]}
