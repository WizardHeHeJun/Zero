"""SupervisorAgent：协调与任务完成节点。

只做协调 + 任务完成判定，**不含业务逻辑**（业务在各 Worker）。记忆写入
**只在此处**发生（节流，见 memory-rules.md #1）：当前情绪事件写 session 作用域，
长期情绪倾向写 user 作用域，均显式 scope。注入 MemoryClient，不直连图谱。

存储边界（与 main.py ConversationLog 并行、职责不重叠）：
- SupervisorAgent 只写情感事件（write，session scope）、长期 episode（write_episode，
  user scope）、disposition（write，user scope）至 MemoryClient。
- 对话 transcript 与 attitude 短期态属运行态，由 main.py ConversationLog 管理
  （turns 表 + meta 表，SQLite，无需 MemoryClient）。
- 两套存储并行运行——不在此处读写对话历史，不在 ConversationLog 写情感记忆。
"""

from __future__ import annotations

from src.agents.affect_math import text_label
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
            # 富 episode：自然语言情感事件 → 语义记忆（Graphiti 抽实体/关系入图）。
            # 无语义后端时 no-op（零回归）；仍只在本任务完成节点写（节流，memory-rules #1）。
            label = text_label(affect[0], affect[1])
            await self.memory.write_episode(
                f"用户对刺激「{stim_name}」表现出 {label} 情绪"
                f"（valence={affect[0]:.2f}, arousal={affect[1]:.2f}），价值估计 {value:.3f}。",
                scope=Scope.USER,
                key=state.user_id,
            )
        entry = {"node": "supervisor", "task_complete": True}
        return {"task_complete": True, "trace": [entry]}
