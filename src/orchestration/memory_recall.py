"""MemoryRecallAgent：长期情绪倾向回灌（记忆读闭环）。

在管线开头读 `user` 作用域的长期情绪倾向（由 Supervisor 在任务完成时写入），
解析出标量 disposition 放进 state，供 AppraisalAgent 偏置先验——让记忆层真正被
『用上』（此前仅写不读，闭合 读↔写 回路）。注入 MemoryClient，不直连图谱。
`recall_enabled` 关闭或无记忆时为 no-op（严格零回归）。
节点契约：(state) -> dict，只返回增量。
"""

from __future__ import annotations

from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.orchestration.state import AffectState


def _parse_disposition(facts: list[Fact]) -> float | None:
    """从最新一条 disposition 事实解析 value 标量（写入格式见 SupervisorAgent）。"""
    if not facts:
        return None
    content = facts[-1].content  # "disposition stimulus=<name> value=<float>"
    marker = "value="
    idx = content.rfind(marker)
    if idx < 0:
        return None
    try:
        return float(content[idx + len(marker) :].split()[0])
    except ValueError:
        return None


class MemoryRecallAgent:
    """读 user 长期倾向 → recalled_disposition（偏置 appraisal）。注入 client，不直连图谱。"""

    def __init__(self, memory: MemoryClient) -> None:
        self.memory = memory

    async def __call__(self, state: AffectState) -> dict:
        if not state.recall_enabled:
            return {}
        facts = await self.memory.query("disposition", scope=Scope.USER, key=state.user_id)
        disposition = _parse_disposition(facts)
        if disposition is None:
            return {}
        entry = {"node": "memory_recall", "recalled_disposition": disposition}
        return {"recalled_disposition": disposition, "trace": [entry]}
