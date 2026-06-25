"""确定性长期记忆图谱后端（带时序失效语义）。

接口对齐 Graphiti：新事实使同 (scope, key) 的旧事实**失效**（设 invalid_at）而非物理删除；
查询带时间语境。三实现同语义：`InMemoryGraphStore`（默认/零依赖）/ `SqliteGraphStore`
（落盘、跨进程/重启留存）/ `Neo4jGraphStore`（裸 Cypher）。工厂 `build_graph_store`
在 `src/storage/graph_store.py`（与其探测 `_neo4j_store` 同模块，便于测试 monkeypatch）。
存储层为最底层，不上调记忆/编排层。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

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
    容器化部署时路径由 env 注入；可经 build_graph_store 切 Neo4j（已实现）或后续 Graphiti。
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


class Neo4jGraphStore:
    """Neo4j 落盘图谱存储；与 InMemory/Sqlite 同语义（时序失效），用裸 Cypher 实现。

    新事实使同 (scope, key) 的旧事实失效（设 invalid_at）而非物理删除；查询带时间语境。
    valid_at/invalid_at 以 ISO 字符串存储、Python 侧按时间语境过滤——与 SqliteGraphStore 同形，
    便于在两后端间无缝切换。需要实体抽取/向量检索时可整体换 Graphiti（见 build_graph_store）。
    驱动惰性连接：构造不发起连接，真正连接发生在首次会话；缺 neo4j 驱动由工厂捕获回退。
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase  # 延迟导入：缺驱动时由 build_graph_store 捕获回退

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """关闭驱动连接池。"""
        self.driver.close()

    def add_fact(self, scope: str, key: str, content: str, valid_at: datetime) -> None:
        """写入一条事实；使同 (scope, key) 的旧事实在 valid_at 失效。"""
        ts = valid_at.isoformat()
        with self.driver.session() as session:
            session.execute_write(self._add_fact_tx, scope, key, content, ts)
        logger.debug("neo4j_graph_store add_fact scope=%s key=%s", scope, key)

    @staticmethod
    def _add_fact_tx(tx: Any, scope: str, key: str, content: str, ts: str) -> None:
        tx.run(
            "MATCH (f:Fact {scope: $scope, key: $key}) "
            "WHERE f.invalid_at IS NULL SET f.invalid_at = $ts",
            scope=scope,
            key=key,
            ts=ts,
        )
        tx.run(
            "CREATE (f:Fact {scope: $scope, key: $key, content: $content, "
            "valid_at: $ts, invalid_at: null})",
            scope=scope,
            key=key,
            content=content,
            ts=ts,
        )

    def query_facts(
        self, scope: str, key: str | None = None, at: datetime | None = None
    ) -> list[StoredFact]:
        """查询在时刻 at 有效的事实（at 为空表示取当前有效事实）。"""
        with self.driver.session() as session:
            rows = session.execute_read(self._query_facts_tx, scope)
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

    @staticmethod
    def _query_facts_tx(tx: Any, scope: str) -> list[tuple[str, str, str, str, str | None]]:
        result = tx.run(
            "MATCH (f:Fact {scope: $scope}) "
            "RETURN f.scope, f.key, f.content, f.valid_at, f.invalid_at",
            scope=scope,
        )
        return [tuple(record.values()) for record in result]
