"""存储层图谱后端**门面**（稳定导入点）。

实现已拆到 `src/storage/backends/`：确定性图谱（`deterministic.py`：InMemory/Sqlite/Neo4j）
与语义侧信道（`semantic.py`：Graphiti/SqliteVector + 助手）。本模块再导出它们，并持有
**工厂**（`build_graph_store`/`build_semantic_store`）及其后端探测（`_neo4j_store`/
`_graphiti_store`/`_sqlite_vector_store`）。

工厂与其探测函数刻意同处本模块：测试用 `monkeypatch.setattr(graph_store, "_neo4j_store", …)`
后调 `build_graph_store()`，二者必须同命名空间方能命中（拆到子模块会令 monkeypatch 失效）。
上层（记忆层/编排层）一律从本模块导入，不直连 `backends/`——便于后续增删后端。
存储层为最底层，不上调记忆/编排层。
"""

from __future__ import annotations

import logging
import os

from src.storage.backends.deterministic import (
    GraphStore,
    InMemoryGraphStore,
    Neo4jGraphStore,
    SqliteGraphStore,
    StoredFact,
)
from src.storage.backends.semantic import (
    GraphitiGraphStore,
    SemanticStore,
    SqliteVectorStore,
    coerce_dt,
    cosine,
    group_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GraphStore",
    "GraphitiGraphStore",
    "InMemoryGraphStore",
    "Neo4jGraphStore",
    "SemanticStore",
    "SqliteGraphStore",
    "SqliteVectorStore",
    "StoredFact",
    "build_graph_store",
    "build_semantic_store",
    "coerce_dt",
    "cosine",
    "group_id",
]


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
        logger.warning("Graphiti 语义后端构造失败（%s），回退无语义记忆", exc, exc_info=True)
        return None


def _sqlite_vector_store() -> SqliteVectorStore | None:
    """构造轻量 SQLite 向量后端；缺 openai（embedding 依赖）时告警回退 None。

    落盘路径 env `ZERO_SEMANTIC_DB`（默认 `data/semantic.sqlite3`，自动建目录；`:memory:` 不落盘）。
    所有 env 读取集中在此工厂，SqliteVectorStore 本身通过显式参数接收配置。
    """
    try:
        import openai  # noqa: F401  仅探测 embedding 依赖是否可用
    except ImportError:
        logger.warning("ZERO_SEMANTIC_BACKEND=sqlite_vec 但缺 openai，回退（无语义记忆）")
        return None
    path = os.getenv("ZERO_SEMANTIC_DB", "data/semantic.sqlite3")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    api_key = os.getenv("ZERO_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("ZERO_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("ZERO_GRAPHITI_EMBED_MODEL", "text-embedding-3-small")
    sim_threshold = float(os.getenv("ZERO_RECALL_SIM_MIN", "0.65"))
    dedup_max = float(os.getenv("ZERO_EPISODE_DEDUP_MAX", "0.92"))
    max_per_key = int(os.getenv("ZERO_EPISODE_MAX_PER_KEY", "0"))
    return SqliteVectorStore(
        path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        sim_threshold=sim_threshold,
        dedup_max=dedup_max,
        max_per_key=max_per_key,
    )


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
