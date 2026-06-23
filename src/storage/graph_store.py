"""长期记忆图谱后端（第一版内存占位），带时序失效语义。

接口对齐 Graphiti：新事实使同 (scope, key) 的旧事实**失效**（设 invalid_at）
而非物理删除；查询带时间语境。记忆层经此读写，编排层不得直连本模块。
存储层为最底层，不上调记忆/编排层。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class StoredFact:
    """图谱中一条带时间维度的事实记录。"""

    scope: str
    key: str
    content: str
    valid_at: datetime
    invalid_at: datetime | None = None


class GraphStore(Protocol):
    """长期记忆图谱后端协议。记忆层依赖本协议而非具体实现，便于替换 Graphiti 等后端。"""

    def add_fact(self, scope: str, key: str, content: str, valid_at: datetime) -> None: ...

    def query_facts(
        self, scope: str, key: str | None = None, at: datetime | None = None
    ) -> list[StoredFact]: ...


class InMemoryGraphStore:
    """内存版图谱存储；演示时序失效，不做真实实体/关系抽取。"""

    def __init__(self) -> None:
        self.facts: list[StoredFact] = []

    def add_fact(self, scope: str, key: str, content: str, valid_at: datetime) -> None:
        """写入一条事实；使同 (scope, key) 的旧事实在 valid_at 失效。"""
        for fact in self.facts:
            if fact.scope == scope and fact.key == key and fact.invalid_at is None:
                fact.invalid_at = valid_at
        self.facts.append(StoredFact(scope=scope, key=key, content=content, valid_at=valid_at))
        logger.debug("graph_store add_fact scope=%s key=%s", scope, key)

    def query_facts(
        self, scope: str, key: str | None = None, at: datetime | None = None
    ) -> list[StoredFact]:
        """查询在时刻 at 有效的事实（at 为空表示取当前有效事实）。"""
        results: list[StoredFact] = []
        for fact in self.facts:
            if fact.scope != scope:
                continue
            if key is not None and fact.key != key:
                continue
            if at is None:
                if fact.invalid_at is None:
                    results.append(fact)
            elif fact.valid_at <= at and (fact.invalid_at is None or at < fact.invalid_at):
                results.append(fact)
        return results
