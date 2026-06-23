"""记忆层读写 API：显式 scope、任务完成节流、封装图谱后端。

上层（编排层/Agent）只能经本 API 访问长期记忆，不得直连图谱/向量库。
写入仅应在 Supervisor 判定「任务完成」的节点调用（节流）；不要在每个
Worker 每步回调里写（见 memory-rules.md #1、pitfalls）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.memory.types import Fact, Scope
from src.storage.graph_store import GraphStore, InMemoryGraphStore

logger = logging.getLogger(__name__)


class MemoryClient:
    """长期记忆读写入口；对下依赖 GraphStore 协议（非具体类），对上屏蔽存储细节。"""

    def __init__(self, store: GraphStore | None = None) -> None:
        self.store: GraphStore = store if store is not None else InMemoryGraphStore()

    async def write(
        self,
        content: str,
        *,
        scope: Scope,
        key: str = "default",
        valid_at: datetime | None = None,
    ) -> None:
        """写入一条记忆。必须显式 scope；仅在任务完成节点调用（节流）。"""
        if not isinstance(scope, Scope):
            raise ValueError("memory.write 必须显式指定 Scope，禁止默认作用域")
        when = valid_at if valid_at is not None else datetime.now(UTC)
        self.store.add_fact(scope=scope.value, key=key, content=content, valid_at=when)
        logger.info("memory.write scope=%s key=%s", scope.value, key)

    async def query(
        self,
        query: str,
        *,
        scope: Scope,
        key: str | None = None,
        at: datetime | None = None,
    ) -> list[Fact]:
        """按作用域与时间语境读取记忆。必须显式 scope；带时间语境处理时序失效。"""
        if not isinstance(scope, Scope):
            raise ValueError("memory.query 必须显式指定 Scope，禁止默认作用域")
        logger.debug("memory.query scope=%s key=%s q=%s", scope.value, key, query)
        stored = self.store.query_facts(scope=scope.value, key=key, at=at)
        return [
            Fact(content=s.content, scope=scope, valid_at=s.valid_at, key=s.key) for s in stored
        ]
