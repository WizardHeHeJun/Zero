"""长期记忆图谱后端（第一版内存占位），带时序失效语义。

接口对齐 Graphiti：新事实使同 (scope, key) 的旧事实**失效**（设 invalid_at）
而非物理删除；查询带时间语境。记忆层经此读写，编排层不得直连本模块。
存储层为最底层，不上调记忆/编排层。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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


def _neo4j_store() -> Neo4jGraphStore | None:
    """构造 Neo4j 后端；缺 neo4j 驱动时告警并返回 None 触发回退（与 checkpointer 同款）。

    连接参数取 env：`ZERO_NEO4J_URI`（默认 `bolt://localhost:7687`）、
    `ZERO_NEO4J_USER`（默认 `neo4j`）、`ZERO_NEO4J_PASSWORD`（默认 `password`，与 compose 一致）。
    """
    try:
        import neo4j  # noqa: F401  仅探测驱动是否可用
    except ImportError:
        logger.warning("ZERO_MEMORY_BACKEND=neo4j 但缺 neo4j 驱动，回退 InMemory")
        return None
    uri = os.getenv("ZERO_NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("ZERO_NEO4J_USER", "neo4j")
    password = os.getenv("ZERO_NEO4J_PASSWORD", "password")
    return Neo4jGraphStore(uri, user, password)


def build_graph_store(backend: str | None = None) -> GraphStore:
    """按 env `ZERO_MEMORY_BACKEND` 选长期记忆后端：`memory`（默认，零回归）/ `sqlite` / `neo4j`。

    `sqlite` 路径取 env `ZERO_GRAPH_DB`（默认 `data/graph.sqlite3`，自动建目录）。
    `neo4j` 连接参数见 `_neo4j_store`；缺驱动告警回退 InMemory。
    容器化部署时通过 env 切后端/路径；需实体抽取/向量检索时可整体换 Graphiti。
    """
    choice = (backend or os.getenv("ZERO_MEMORY_BACKEND") or "memory").lower()
    if choice == "sqlite":
        path = os.getenv("ZERO_GRAPH_DB", "data/graph.sqlite3")
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return SqliteGraphStore(path)
    if choice == "neo4j":
        store = _neo4j_store()
        if store is not None:
            return store
        return InMemoryGraphStore()
    return InMemoryGraphStore()


# --------------------------------------------------------------------------- #
# 语义记忆侧信道（Graphiti）：与上面的确定性 GraphStore 并存、互不影响。
# 确定性 (scope,key) 失效模型走 GraphStore；富 episode 写入 + 语义/向量召回走
# 下面的 SemanticStore。默认无后端（build_semantic_store -> None），严格零回归。
# --------------------------------------------------------------------------- #


@runtime_checkable
class SemanticStore(Protocol):
    """语义记忆后端协议（Graphiti 等）：富 episode 写入 + 语义召回，全异步。

    与 `GraphStore`（确定性 (scope,key) 失效）正交：记忆层据能力检测择优，
    无实现时上层自动 no-op。`search` 用 query 文本做语义/向量检索（GraphStore
    的 query_facts 不吃 query，二者语义不同）。
    """

    async def add_episode(
        self, *, scope: str, key: str, content: str, valid_at: datetime
    ) -> None: ...

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list[StoredFact]: ...


class GraphitiGraphStore:
    """Graphiti 语义记忆后端：LLM 抽取实体/关系入图 + 语义向量检索 + 时序失效。

    scope/key → Graphiti `group_id`（`f"{scope}:{key}"`）。`add_episode` 写自然语言
    episode（Graphiti 自动抽实体/关系）；`search` 做语义检索，边 `.fact/.valid_at/
    .invalid_at` 映射成 `StoredFact` 并按 `at` 做时间语境过滤。失效语义由 Graphiti
    （LLM/矛盾驱动）负责，与 GraphStore 的确定性同 key 覆盖不同——故只验往返、不强断言。

    构造不连接：`Graphiti(...)` 仅建驱动；首次读写时一次性建索引/约束（flag + Lock）。
    LLM/embedder 复用语言层 env（`ZERO_OPENAI_*`，回退 `OPENAI_*`）+ `ZERO_GRAPHITI_MODEL`；
    Neo4j 连接复用 `ZERO_NEO4J_*`。驱动惰性导入：缺 graphiti-core 由工厂捕获回退。
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.graphiti = _build_graphiti(uri, user, password)
        self.initialized = False
        self.init_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        """首次读写时一次性建索引/约束（双检锁，避免并发重复 setup）。"""
        if self.initialized:
            return
        async with self.init_lock:
            if self.initialized:
                return
            await self.graphiti.build_indices_and_constraints()
            self.initialized = True

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        """写一条自然语言 episode；Graphiti 抽取实体/关系入图，按 group_id 隔离。"""
        from graphiti_core.nodes import EpisodeType  # 延迟导入：缺驱动由工厂回退

        await self._ensure_init()
        group_id = _group_id(scope, key)
        await self.graphiti.add_episode(
            name=group_id,
            episode_body=content,
            source=EpisodeType.text,
            source_description="affective-expression memory",
            reference_time=valid_at,
            group_id=group_id,
        )
        logger.debug("graphiti add_episode scope=%s key=%s", scope, key)

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list[StoredFact]:
        """语义检索 group_id 内事实，映射成 StoredFact 并按时间语境过滤。"""
        await self._ensure_init()
        group_id = _group_id(scope, key)
        edges = await self.graphiti.search(query, group_ids=[group_id])
        results: list[StoredFact] = []
        for edge in edges:
            valid = _coerce_dt(getattr(edge, "valid_at", None))
            invalid = _coerce_dt(getattr(edge, "invalid_at", None))
            if at is None:
                if invalid is not None:
                    continue
            elif (valid is not None and valid > at) or (invalid is not None and at >= invalid):
                continue
            results.append(
                StoredFact(
                    scope=scope,
                    key=key or "",
                    content=edge.fact,
                    valid_at=valid or at or datetime.now(UTC),
                    invalid_at=invalid,
                )
            )
            if len(results) >= limit:
                break
        return results

    async def close(self) -> None:
        """关闭 Graphiti（底层图库驱动：neo4j 连接池 / kuzu 嵌入式连接）。"""
        await self.graphiti.close()


class SqliteVectorStore:
    """轻量语义记忆后端：SQLite 存 episode 文本+embedding，search 算余弦相似度 Top-K。

    满足 SemanticStore 协议但**不依赖图数据库**——无服务/无 Docker，适合小体量本地原型/验证。
    embedding 走 OpenAI 兼容接口（复用 `ZERO_OPENAI_*` + `ZERO_GRAPHITI_EMBED_MODEL`，默认
    text-embedding-3-small）。丢掉 Graphiti 的实体/关系抽取，只保留语义召回；需要知识图谱时
    换 GraphitiGraphStore（同 SemanticStore 协议，编排层无感切换）。
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            "scope TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL, "
            "valid_at TEXT NOT NULL, embedding TEXT NOT NULL)"
        )
        self.conn.commit()
        self.client: Any = None
        self.base_url = os.getenv("ZERO_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("ZERO_GRAPHITI_EMBED_MODEL", "text-embedding-3-small")

    async def _embed(self, text: str) -> list[float]:
        """文本 → embedding 向量（OpenAI 兼容接口，延迟建 client；缺 openai 由工厂回退）。"""
        if self.client is None:
            from openai import AsyncOpenAI  # 延迟导入

            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = await self.client.embeddings.create(model=self.model, input=[text])
        return list(resp.data[0].embedding)

    async def add_episode(
        self, *, scope: str, key: str, content: str, valid_at: datetime
    ) -> None:
        """写一条 episode：存文本 + 其 embedding。"""
        emb = await self._embed(content)
        self.conn.execute(
            "INSERT INTO episodes (scope, key, content, valid_at, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (scope, key, content, valid_at.isoformat(), json.dumps(emb)),
        )
        self.conn.commit()
        logger.debug("sqlite_vector add_episode scope=%s key=%s", scope, key)

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list[StoredFact]:
        """对 scope[/key] 下 episode 算 query 余弦相似度，返回 Top-limit（带时间语境过滤）。"""
        q = await self._embed(query)
        rows = self.conn.execute(
            "SELECT scope, key, content, valid_at, embedding FROM episodes WHERE scope = ?",
            (scope,),
        ).fetchall()
        scored: list[tuple[float, StoredFact]] = []
        for s, k, content, valid_at, emb_json in rows:
            if key is not None and k != key:
                continue
            valid = datetime.fromisoformat(valid_at)
            if at is not None and valid > at:
                continue
            sim = _cosine(q, json.loads(emb_json))
            scored.append((sim, StoredFact(scope=s, key=k, content=content, valid_at=valid)))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        self.conn.close()


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（纯 Python，不依赖 numpy）；任一零向量返回 0。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def _build_graphiti(uri: str, user: str, password: str) -> Any:
    """构造 Graphiti 客户端：env `ZERO_GRAPHITI_DB` 选图库 neo4j（默认）/ kuzu（嵌入式，无服务）。

    LLM/embedder：显式配 OpenAI 兼容网关（`ZERO_OPENAI_*` / `ZERO_GRAPHITI_MODEL`）则注入，
    否则用 Graphiti 默认（读 `OPENAI_API_KEY`）；自定义构造失败时告警回退默认。
    ⚠ kuzu 后端 upstream 已 deprecated（会被移除），仅作本地 smoke 验证用；持久/生产走 neo4j。
    """
    from graphiti_core import Graphiti  # 延迟导入：缺驱动由 _graphiti_store 捕获回退

    base_url = os.getenv("ZERO_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("ZERO_GRAPHITI_MODEL")
    embed_model = os.getenv("ZERO_GRAPHITI_EMBED_MODEL")  # 网关嵌入模型名（默认 3-small）
    # 兜底：把 ZERO_OPENAI_* 暴露为标准 OPENAI_*，让 Graphiti 内部任何默认 OpenAI 子客户端也拿到凭证
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
    llm_kwargs: dict[str, Any] = {}
    if base_url or model:
        try:
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_client import OpenAIClient

            llm_config = LLMConfig(api_key=api_key, base_url=base_url, model=model)
            embed_cfg: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
            if embed_model:
                embed_cfg["embedding_model"] = embed_model
            llm_kwargs = {
                "llm_client": OpenAIClient(config=llm_config),
                "embedder": OpenAIEmbedder(config=OpenAIEmbedderConfig(**embed_cfg)),
                # 显式传重排客户端，否则 Graphiti 默认 `OpenAIRerankerClient()` 无参构造、
                # 从标准 `OPENAI_API_KEY` 读凭证 → 缺凭证报错（key 在 ZERO_OPENAI_API_KEY）
                "cross_encoder": OpenAIRerankerClient(config=llm_config),
            }
        except Exception:
            logger.warning("自定义 Graphiti LLM/embedder 构造失败，回退默认 OpenAI 配置")

    if (os.getenv("ZERO_GRAPHITI_DB") or "neo4j").lower() == "kuzu":
        from graphiti_core.driver.kuzu_driver import KuzuDriver  # 嵌入式：缺 kuzu 由工厂回退

        path = os.getenv("ZERO_KUZU_PATH", "data/graphiti.kuzu")
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return Graphiti(graph_driver=KuzuDriver(db=path), **llm_kwargs)
    return Graphiti(uri, user, password, **llm_kwargs)


def _coerce_dt(value: Any) -> datetime | None:
    """把 Graphiti 边的 valid_at/invalid_at 归一成 datetime|None。

    防 Kuzu 后端已知 bug（getzep/graphiti #893：valid_at 可能回非 datetime/字符串格式）——
    无法解析则当 None（按"无界/当前有效"处理），避免 `str > datetime` 抛错。
    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _group_id(scope: str, key: str | None) -> str:
    """构造 Graphiti group_id：`scope[_key]`，并把非 [A-Za-z0-9_-] 字符替换为 `_`。

    Graphiti 的 validate_group_id 只允许字母数字/横杠/下划线——冒号等会报
    GroupIdValidationError，故不能用 `f"{scope}:{key}"`。读写须用同一构造保证可召回。
    """
    raw = scope if key is None else f"{scope}_{key}"
    return "".join(c if (c.isascii() and (c.isalnum() or c in "_-")) else "_" for c in raw)


def _graphiti_store() -> GraphitiGraphStore | None:
    """构造 Graphiti 语义后端；缺驱动/构造失败时告警回退 None（语义是可选侧信道，绝不拖垮主管线）。

    图库经 env `ZERO_GRAPHITI_DB` 选：`neo4j`（默认，复用 `ZERO_NEO4J_{URI,USER,PASSWORD}`）/
    `kuzu`（嵌入式，落盘 `ZERO_KUZU_PATH`，默认 `data/graphiti.kuzu`，本地无服务）。
    """
    try:
        uri = os.getenv("ZERO_NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("ZERO_NEO4J_USER", "neo4j")
        password = os.getenv("ZERO_NEO4J_PASSWORD", "password")
        return GraphitiGraphStore(uri, user, password)
    except Exception as exc:
        logger.warning("Graphiti 语义后端构造失败（%s），回退无语义记忆", exc)
        return None


def _sqlite_vector_store() -> SqliteVectorStore | None:
    """构造轻量 SQLite 向量后端；缺 openai（embedding 依赖）时告警回退 None。

    落盘路径 env `ZERO_SEMANTIC_DB`（默认 `data/semantic.sqlite3`，自动建目录；`:memory:` 不落盘）。
    """
    try:
        import openai  # noqa: F401  仅探测 embedding 依赖是否可用
    except ImportError:
        logger.warning("ZERO_SEMANTIC_BACKEND=sqlite_vec 但缺 openai，回退（无语义记忆）")
        return None
    path = os.getenv("ZERO_SEMANTIC_DB", "data/semantic.sqlite3")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return SqliteVectorStore(path)


def build_semantic_store(backend: str | None = None) -> SemanticStore | None:
    """按 env `ZERO_SEMANTIC_BACKEND` 选语义后端：空（默认/零回归）/ `sqlite_vec` / `graphiti`。

    与 `build_graph_store`（确定性后端）正交：语义记忆是可选侧信道，默认不启用、零依赖。
    `sqlite_vec`（轻量、无图库、SQLite+向量相似度）缺 openai、`graphiti` 缺驱动 → 告警回退 None。
    """
    choice = (backend or os.getenv("ZERO_SEMANTIC_BACKEND") or "").lower()
    if choice == "graphiti":
        return _graphiti_store()
    if choice in ("sqlite_vec", "sqlite", "vector"):
        return _sqlite_vector_store()
    return None
