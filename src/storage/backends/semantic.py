"""语义记忆侧信道后端（Graphiti / 轻量 SQLite 向量）。

与 deterministic 的确定性 GraphStore（(scope,key) 失效）正交并存：富 episode 写入 +
语义/向量召回走本模块。两实现满足 `SemanticStore` 协议：`GraphitiGraphStore`（LLM 抽
实体/关系入图 + 语义检索，需图库）/ `SqliteVectorStore`（SQLite + 余弦相似度，无图库/无服务）。
工厂 `build_semantic_store` 在 `src/storage/graph_store.py`（与其探测 `_graphiti_store`/
`_sqlite_vector_store` 同模块，便于测试 monkeypatch）；默认无后端（None）、严格零回归。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from src.storage.backends.deterministic import StoredFact

logger = logging.getLogger(__name__)


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
        sim_threshold: float | None = None,
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
        sim_threshold: float | None = None,
    ) -> list[StoredFact]:
        """语义检索 group_id 内事实，映射成 StoredFact 并按时间语境过滤。

        sim_threshold 参数保持协议兼容（Graphiti 侧暂不实现过滤，图检索分数由 Graphiti 内部管理）。
        """
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
        self.sim_threshold = float(os.getenv("ZERO_RECALL_SIM_MIN", "0.65"))
        self.dedup_threshold = float(os.getenv("ZERO_EPISODE_DEDUP_MAX", "0.92"))

    async def _embed(self, text: str) -> list[float]:
        """文本 → embedding 向量（OpenAI 兼容接口，延迟建 client；缺 openai 由工厂回退）。"""
        try:
            if self.client is None:
                from openai import AsyncOpenAI  # 延迟导入

                self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            resp = await self.client.embeddings.create(model=self.model, input=[text])
            return list(resp.data[0].embedding)
        except Exception:
            logger.debug("embedding 调用失败 model=%s", self.model, exc_info=True)
            raise

    async def add_episode(self, *, scope: str, key: str, content: str, valid_at: datetime) -> None:
        """写一条 episode：存文本 + 其 embedding。写入前做 dedup 检测，跳过高度相似的重复内容。"""
        emb = await self._embed(content)
        # B-5 dedup：不走 search 的 sim_threshold 剪枝，直接比对所有已存 episode 原始向量
        try:
            rows = self.conn.execute(
                "SELECT embedding FROM episodes WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchall()
            for (emb_json,) in rows:
                existing_emb = json.loads(emb_json)
                if _cosine(emb, existing_emb) > self.dedup_threshold:
                    logger.debug(
                        "sqlite_vector dedup skip scope=%s key=%s (sim>%.2f)",
                        scope,
                        key,
                        self.dedup_threshold,
                    )
                    return
        except Exception as exc:
            # dedup 失败保守退化：正常写入，宁多写不崩
            logger.warning(
                "dedup probe failed scope=%s key=%s: %s, writing anyway",
                scope,
                key,
                exc,
                exc_info=True,
            )
        self.conn.execute(
            "INSERT INTO episodes (scope, key, content, valid_at, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (scope, key, content, valid_at.isoformat(), json.dumps(emb)),
        )
        self.conn.commit()
        # D7 容量上限：写后剪裁，保留同 (scope,key) 下按时间最新 N 条、删超量最旧。
        # 默认 ZERO_EPISODE_MAX_PER_KEY=0 不限（零回归）；仅本写路径执行（守 BLOCK-C）。
        self._trim_capacity(scope, key)
        logger.debug("sqlite_vector add_episode scope=%s key=%s", scope, key)

    def _trim_capacity(self, scope: str, key: str) -> None:
        """按 ZERO_EPISODE_MAX_PER_KEY 删除同 (scope,key) 超量最旧 episode（0=不限）。

        保留按 (valid_at, rowid) 最新 N 条；valid_at 相同则后插入（rowid 更大）者更新。
        容量管理（非时序失效语义），仅在 add_episode 写后调用，绝不在 search/读路径触发。
        """
        max_per_key = int(os.getenv("ZERO_EPISODE_MAX_PER_KEY", "0"))
        if max_per_key <= 0:
            return
        self.conn.execute(
            "DELETE FROM episodes WHERE scope = ? AND key = ? AND rowid NOT IN ("
            "SELECT rowid FROM episodes WHERE scope = ? AND key = ? "
            "ORDER BY valid_at DESC, rowid DESC LIMIT ?)",
            (scope, key, scope, key, max_per_key),
        )
        self.conn.commit()

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
        sim_threshold: float | None = None,
    ) -> list[StoredFact]:
        """对 scope[/key] 下 episode 算 query 余弦相似度，返回 Top-limit（带时间语境过滤）。

        sim_threshold 显式传时使用该值，否则用 self.sim_threshold（来自 env
        ZERO_RECALL_SIM_MIN，默认 0.65）过滤低相似度结果。
        """
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
            scored.append(
                (sim, StoredFact(scope=s, key=k, content=content, valid_at=valid, sim=sim))
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        threshold = sim_threshold if sim_threshold is not None else self.sim_threshold
        return [fact for sim, fact in scored[:limit] if sim >= threshold]

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
            logger.warning(
                "自定义 Graphiti LLM/embedder 构造失败，回退默认 OpenAI 配置", exc_info=True
            )

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
