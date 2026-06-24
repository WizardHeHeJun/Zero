"""长期记忆图谱后端（第一版内存占位），带时序失效语义。

接口对齐 Graphiti：新事实使同 (scope, key) 的旧事实**失效**（设 invalid_at）
而非物理删除；查询带时间语境。记忆层经此读写，编排层不得直连本模块。
存储层为最底层，不上调记忆/编排层。
"""

from __future__ import annotations

import logging
import os
import sqlite3
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


class SqliteGraphStore:
    """SQLite 落盘图谱存储；与 InMemoryGraphStore 同语义但持久化、可跨进程/重启留存。

    新事实使同 (scope, key) 的旧事实失效（设 invalid_at）而非物理删除；查询带时间语境。
    容器化部署时路径由 env 注入；后续可整体换 Neo4j/Graphiti（见 build_graph_store）。
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "scope TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL, "
            "valid_at TEXT NOT NULL, invalid_at TEXT)"
        )
        self.conn.commit()

    def add_fact(self, scope: str, key: str, content: str, valid_at: datetime) -> None:
        """写入一条事实；使同 (scope, key) 的旧事实在 valid_at 失效。"""
        ts = valid_at.isoformat()
        self.conn.execute(
            "UPDATE facts SET invalid_at = ? WHERE scope = ? AND key = ? AND invalid_at IS NULL",
            (ts, scope, key),
        )
        self.conn.execute(
            "INSERT INTO facts (scope, key, content, valid_at, invalid_at) "
            "VALUES (?, ?, ?, ?, NULL)",
            (scope, key, content, ts),
        )
        self.conn.commit()
        logger.debug("sqlite_graph_store add_fact scope=%s key=%s", scope, key)

    def query_facts(
        self, scope: str, key: str | None = None, at: datetime | None = None
    ) -> list[StoredFact]:
        """查询在时刻 at 有效的事实（at 为空表示取当前有效事实）。"""
        rows = self.conn.execute(
            "SELECT scope, key, content, valid_at, invalid_at FROM facts WHERE scope = ?",
            (scope,),
        ).fetchall()
        results: list[StoredFact] = []
        for s, k, content, valid_at, invalid_at in rows:
            if key is not None and k != key:
                continue
            valid = datetime.fromisoformat(valid_at)
            invalid = datetime.fromisoformat(invalid_at) if invalid_at else None
            fact = StoredFact(scope=s, key=k, content=content, valid_at=valid, invalid_at=invalid)
            if at is None:
                if invalid is None:
                    results.append(fact)
            elif valid <= at and (invalid is None or at < invalid):
                results.append(fact)
        return results


def build_graph_store(backend: str | None = None) -> GraphStore:
    """按 env `ZERO_MEMORY_BACKEND` 选长期记忆后端：`memory`（默认，零回归）/ `sqlite`。

    `sqlite` 路径取 env `ZERO_GRAPH_DB`（默认 `data/graph.sqlite3`，自动建目录）。
    容器化部署时通过 env 切后端/路径；Neo4j/Graphiti 适配器留待 `db` extra（gated）。
    """
    choice = (backend or os.getenv("ZERO_MEMORY_BACKEND") or "memory").lower()
    if choice == "sqlite":
        path = os.getenv("ZERO_GRAPH_DB", "data/graph.sqlite3")
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return SqliteGraphStore(path)
    return InMemoryGraphStore()
