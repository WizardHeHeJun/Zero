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

import os

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
            # 富 episode：B-3 门控（salience 低于阈值跳过写入）
            # salience = precision * |rpe|（rpe=None 时用 0.5 保守估计）
            salience = (state.affect_precision or 0.0) * (
                abs(state.rpe) if state.rpe is not None else 0.5
            )
            ep_threshold = float(os.getenv("ZERO_EPISODE_SALIENCE_MIN", "0.15"))
            if salience >= ep_threshold:
                # B-1 gist：用户原话（stimulus.text）优先，退化到 name[:40]
                user_text = (state.stimulus.text or "") if state.stimulus is not None else ""
                gist = f"你说：{user_text[:200]}" if user_text else f"话题：{stim_name[:40]}"

                # B-2 language 段：language_text 非空才拼，否则省略（language_enabled=False 干净）
                lang_seg = ""
                if state.language_text:
                    lang_seg = f" / 我说：{(state.language_text or '')[:200]}"

                # 情绪标签 + 坐标
                label = text_label(affect[0], affect[1])
                streams = state.ignited_streams or []

                episode_content = (
                    f"{gist}{lang_seg}"
                    f" | 情绪={label}({affect[0]:.2f},{affect[1]:.2f})"
                    f" | precision={state.affect_precision or 0.0:.2f}"
                    f" | streams={streams}"
                    f" | value={value:.3f}"
                )
                # 无语义后端时 no-op（零回归）；仍只在本任务完成节点写（节流，memory-rules #1）
                await self.memory.write_episode(
                    episode_content,
                    scope=Scope.USER,
                    key=state.user_id,
                )
        entry = {"node": "supervisor", "task_complete": True}
        return {"task_complete": True, "trace": [entry]}
