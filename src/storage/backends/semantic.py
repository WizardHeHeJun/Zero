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
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
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

    async def add_episode(
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
    ) -> None:
        """写一条自然语言 episode；Graphiti 抽取实体/关系入图，按 group_id 隔离。

        `embed_text` 仅对自管向量的后端（SqliteVector）有意义；Graphiti 自行从 content 抽取
        实体/关系与向量，故此处忽略该参数（保协议兼容）。
        """
        from graphiti_core.nodes import EpisodeType  # 延迟导入：缺驱动由工厂回退

        await self._ensure_init()
        gid = group_id(scope, key)
        await self.graphiti.add_episode(
            name=gid,
            episode_body=content,
            source=EpisodeType.text,
            source_description="affective-expression memory",
            reference_time=valid_at,
            group_id=gid,
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
        gid = group_id(scope, key)
        edges = await self.graphiti.search(query, group_ids=[gid])
        results: list[StoredFact] = []
        for edge in edges:
            valid = coerce_dt(getattr(edge, "valid_at", None))
            invalid = coerce_dt(getattr(edge, "invalid_at", None))
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

    并发安全：async 方法内的 SQLite I/O 经 asyncio.to_thread 桥接到线程池，避免阻塞事件循环。
    conn 以 check_same_thread=False 建立（线程池各线程共享同一连接），_db_lock（asyncio.Lock）
    串行化写段，防止并发 await 时多线程同时写同一连接。

    构造参数（均可由 build_semantic_store 工厂从 env 传入，存储层本身不直读 env）：
      path          SQLite 落盘路径（默认 ":memory:"）
      api_key       OpenAI 兼容接口密钥
      base_url      OpenAI 兼容接口 base URL
      model         embedding 模型名（默认 "text-embedding-3-small"）
      sim_threshold search 余弦过滤下限（默认 0.65）
      dedup_max     写入 dedup 相似度上限（默认 0.92）
      max_per_key   同 (scope,key) 下 episode 最大保留数（0=不限，默认 0=零回归）
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        sim_threshold: float = 0.65,
        dedup_max: float = 0.92,
        max_per_key: int | None = None,
    ) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            "scope TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL, "
            "valid_at TEXT NOT NULL, embedding TEXT NOT NULL, "
            "access_count INTEGER DEFAULT 0, last_accessed TEXT, "
            "consolidation_count INTEGER DEFAULT 0, "
            "decay_weight REAL DEFAULT 1.0, invalid_at TEXT)"
        )
        # 旧库兼容迁移：新增 5 列（已有列跳过 OperationalError：duplicate column name）。
        # 新库由 CREATE TABLE 直接含 10 列；此段对新库无害（try/except 吞掉已有列错误）。
        for _col_ddl in (
            "ALTER TABLE episodes ADD COLUMN access_count INTEGER DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN last_accessed TEXT",
            "ALTER TABLE episodes ADD COLUMN consolidation_count INTEGER DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN decay_weight REAL DEFAULT 1.0",
            "ALTER TABLE episodes ADD COLUMN invalid_at TEXT",
        ):
            try:
                self.conn.execute(_col_ddl)
            except sqlite3.OperationalError:
                pass  # 列已存在，跳过
        self.conn.commit()
        self.db_lock = asyncio.Lock()
        self.client: Any = None
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.sim_threshold = sim_threshold
        self.dedup_threshold = dedup_max
        # max_per_key：显式传入时用传入值；否则构造期读一次 env（固化，不在热路径重读）。
        # 工厂（_sqlite_vector_store）从 env 读后显式传入；直接构造时（测试/单用）此处兜底。
        # 默认 0=不限=零回归。
        if max_per_key is None:
            self.max_per_key: int = int(os.getenv("ZERO_EPISODE_MAX_PER_KEY", "0"))
        else:
            self.max_per_key = max_per_key

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

    async def add_episode(
        self,
        *,
        scope: str,
        key: str,
        content: str,
        valid_at: datetime,
        embed_text: str | None = None,
    ) -> None:
        """写一条 episode：存 `content` 全文 + 对 `embed_text`（缺省=content）的 embedding。

        议会 A（召回桥根治）：写入端把**用于检索的语义 gist**（embed_text，如「你说：…/我说：…」）
        与**存储/展示全文**（content，含 情绪/precision/streams/value 等结构化元数据）分离——
        否则元数据数字会稀释向量，使「下午两点」这类细节按句意检索时相似度被拉低（encoding
        specificity，Tulving 1973）。`embed_text=None` 时退化为对全文嵌入（旧行为，零回归）。

        SQLite I/O 经 asyncio.to_thread 桥接，不阻塞事件循环；db_lock 串行化并发写段。
        """
        emb = await self._embed(embed_text or content)

        # B-5 dedup + INSERT 合并到同一 db_lock 段（WARN-6 TOCTOU 修复）：
        # 原先 dedup 读与 INSERT 各持独立锁，中间 await 释放锁后并发写可绕过 dedup。
        # 现在把「dedup 查重 + 插入」放在单次 to_thread 调用内、整段持锁，彻底消竞态窗口。
        conn = self.conn
        dedup_threshold = self.dedup_threshold
        valid_at_iso = valid_at.isoformat()
        emb_json = json.dumps(emb)

        def _sync_dedup_and_insert() -> bool:
            """dedup 检查 + INSERT 原子执行（在同一线程内，不跨 await）。
            返回 True=已插入，False=dedup 命中跳过。
            """
            existing_embs = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT embedding FROM episodes WHERE scope = ? AND key = ?",
                    (scope, key),
                ).fetchall()
            ]
            for existing_emb in existing_embs:
                if cosine(emb, existing_emb) > dedup_threshold:
                    return False
            conn.execute(
                "INSERT INTO episodes "
                "(scope, key, content, valid_at, embedding, "
                "access_count, last_accessed, consolidation_count, "
                "decay_weight, invalid_at) "
                "VALUES (?, ?, ?, ?, ?, 0, NULL, 0, 1.0, NULL)",
                (scope, key, content, valid_at_iso, emb_json),
            )
            conn.commit()
            return True

        try:
            async with self.db_lock:
                inserted = await asyncio.to_thread(_sync_dedup_and_insert)
        except Exception as exc:
            # dedup+插入整段失败保守退化：告警并跳过（不崩管线）
            logger.warning(
                "dedup+insert failed scope=%s key=%s: %s, skipping episode",
                scope,
                key,
                exc,
                exc_info=True,
            )
            return

        if not inserted:
            logger.debug(
                "sqlite_vector dedup skip scope=%s key=%s (sim>%.2f)",
                scope,
                key,
                dedup_threshold,
            )
            return

        # D7 容量上限：写后剪裁，保留同 (scope,key) 下按时间最新 N 条、删超量最旧。
        # 默认 ZERO_EPISODE_MAX_PER_KEY=0 不限（零回归）；仅本写路径执行（守 BLOCK-C）。
        await self._trim_capacity(scope, key)
        logger.debug("sqlite_vector add_episode scope=%s key=%s", scope, key)

    async def _trim_capacity(self, scope: str, key: str) -> None:
        """按 self.max_per_key 删除同 (scope,key) 超量最旧 episode（0=不限，零回归）。

        保留按 (valid_at, rowid) 最新 N 条；valid_at 相同则后插入（rowid 更大）者更新。
        容量管理（非时序失效语义），仅在 add_episode 写后调用，绝不在 search/读路径触发。
        SQLite I/O 经 asyncio.to_thread 桥接；db_lock 由调用方（add_episode）外层已不持有，
        此处重新获取以串行化 DELETE。max_per_key 由工厂（_sqlite_vector_store）从 env 读取后
        经构造参数传入，存储层本身不再直读 env（修复 BLOCK-1）。
        """
        max_per_key = self.max_per_key
        if max_per_key <= 0:
            return

        conn = self.conn

        def _sync_trim() -> None:
            conn.execute(
                "DELETE FROM episodes WHERE scope = ? AND key = ? AND rowid NOT IN ("
                "SELECT rowid FROM episodes WHERE scope = ? AND key = ? "
                # 优先保留高 decay_weight（巩固度高）的 episode；同等则保留最新；
                # rowid 作末位排歧（最后插入=更新=优先保留）。
                # ⚠ 已知缺陷（议会二轮张力 2a·2026-08-13·本批**未修**）：巩固子系统默认关
                # ⇒ decay_weight 恒为建表默认 1.0 ⇒ 第一排序键失效、退化纯 FIFO，最早的
                # 身份自陈会被超容量驱逐（438 轮实测已发生，findings §四点五 成因 B）。
                # 裁定的修法是加 importance 键，但 importance 由记忆层
                # `memory.utils.importance_signal` 从 content 算出——本存储层**不得**反向
                # import 记忆层（红线 1），也不得加 DB 列（议会 D1 BLOCK：两后端能力分叉）
                # ⇒ 需要写入时预计算或注入式回调，属记忆生命周期语义，归入
                # SleepConsolidation「修 vs 退役」独立 PRP 一并裁，勿在此顺手实现。
                "ORDER BY decay_weight DESC, valid_at DESC, rowid DESC LIMIT ?)",
                (scope, key, scope, key, max_per_key),
            )
            conn.commit()

        async with self.db_lock:
            await asyncio.to_thread(_sync_trim)

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

        sim_threshold 显式传时使用该值，否则用 self.sim_threshold（来自构造参数/工厂
        ZERO_RECALL_SIM_MIN，默认 0.65）过滤低相似度结果。
        invalid_at 非 NULL 的行为软删除状态，读路径跳过（巩固迁移后的 session 行）。
        rowid 作 episode_id 透传，供 Supervisor 节流更新 access_count（ACT-R 频率）。
        SQLite I/O 经 asyncio.to_thread 桥接，不阻塞事件循环。
        """
        q = await self._embed(query)
        conn = self.conn

        def _sync_fetch() -> list[tuple]:
            # 读 rowid + 10 列；WHERE 过滤软删除行（invalid_at IS NULL = 有效）
            return conn.execute(
                "SELECT rowid, scope, key, content, valid_at, embedding, "
                "access_count, decay_weight "
                "FROM episodes WHERE scope = ? AND invalid_at IS NULL",
                (scope,),
            ).fetchall()

        async with self.db_lock:
            rows = await asyncio.to_thread(_sync_fetch)

        scored: list[tuple[float, StoredFact]] = []
        for rowid, s, k, content, valid_at_str, emb_json, _ac, _dw in rows:
            if key is not None and k != key:
                continue
            valid = datetime.fromisoformat(valid_at_str)
            if at is not None and valid > at:
                continue
            sim = cosine(q, json.loads(emb_json))
            scored.append(
                (
                    sim,
                    StoredFact(
                        scope=s,
                        key=k,
                        content=content,
                        valid_at=valid,
                        sim=sim,
                        episode_id=str(rowid),
                        access_count=_ac if _ac is not None else 0,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        threshold = sim_threshold if sim_threshold is not None else self.sim_threshold
        return [fact for sim, fact in scored[:limit] if sim >= threshold]

    async def batch_update_access_count(self, episode_ids: list[str]) -> None:
        """节流更新 access_count + last_accessed（ACT-R 频率·仅 Supervisor 任务完成节点调用）。

        episode_ids 为 search 路径透传的 rowid（str）列表；UPDATE 按 rowid 精确命中。
        不在召回节点调用（CS BLOCK：召回时更新 access_count 会污染当轮排序自一致性）。
        SQLite I/O 经 asyncio.to_thread 桥接；db_lock 串行化写段。
        """
        if not episode_ids:
            return
        conn = self.conn
        now_iso = datetime.now(UTC).isoformat()

        def _sync_update() -> None:
            for eid in episode_ids:
                try:
                    conn.execute(
                        "UPDATE episodes SET access_count = access_count + 1, "
                        "last_accessed = ? WHERE rowid = ?",
                        (now_iso, int(eid)),
                    )
                except (ValueError, sqlite3.Error) as exc:
                    logger.warning("batch_update_access_count rowid=%s 失败: %s", eid, exc)
            conn.commit()

        async with self.db_lock:
            await asyncio.to_thread(_sync_update)
        logger.debug("sqlite_vector batch_update_access_count n=%d", len(episode_ids))

    async def apply_decay_weights(self, updates: list[tuple[float, str]]) -> None:
        """批量更新 decay_weight（Ebbinghaus 分层幂律衰减·Consolidation 调用）。

        updates 为 [(new_decay_weight, episode_id), ...]；episode_id = rowid（str）。
        decay_weight 不物理删除——衰减至低值后由 _trim_capacity 容量管理优先驱逐。
        SQLite I/O 经 asyncio.to_thread 桥接；db_lock 串行化写段。
        """
        if not updates:
            return
        conn = self.conn

        def _sync_apply() -> None:
            for dw, eid in updates:
                try:
                    conn.execute(
                        "UPDATE episodes SET decay_weight = ? WHERE rowid = ?",
                        (dw, int(eid)),
                    )
                except (ValueError, sqlite3.Error) as exc:
                    logger.warning("apply_decay_weights rowid=%s 失败: %s", eid, exc)
            conn.commit()

        async with self.db_lock:
            await asyncio.to_thread(_sync_apply)
        logger.debug("sqlite_vector apply_decay_weights n=%d", len(updates))

    async def consolidate_session_to_user(
        self,
        scope_from: str,
        scope_to: str,
        rowids: list[str],
    ) -> None:
        """把 SESSION episode 升迁到 USER scope（睡眠巩固·系统巩固工程近似）。

        流程：复制指定 rowid 行到 scope_to（consolidation_count+1）→ 原行 invalid_at=now 软删。
        不物理删：保留历史溯源；search 路径过滤 invalid_at IS NULL 自动跳过软删行。
        SQLite I/O 经 asyncio.to_thread 桥接；db_lock 串行化写段。
        """
        if not rowids:
            return
        conn = self.conn
        now_iso = datetime.now(UTC).isoformat()

        def _sync_consolidate() -> None:
            for eid in rowids:
                try:
                    rid = int(eid)
                    row = conn.execute(
                        "SELECT key, content, valid_at, embedding, "
                        "access_count, consolidation_count, decay_weight "
                        "FROM episodes WHERE rowid = ?",
                        (rid,),
                    ).fetchone()
                    if row is None:
                        continue
                    key, content, valid_at, emb, ac, cc, dw = row
                    conn.execute(
                        "INSERT INTO episodes "
                        "(scope, key, content, valid_at, embedding, "
                        "access_count, last_accessed, consolidation_count, "
                        "decay_weight, invalid_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)",
                        (
                            scope_to,
                            key,
                            content,
                            valid_at,
                            emb,
                            ac if ac is not None else 0,
                            (cc if cc is not None else 0) + 1,
                            dw if dw is not None else 1.0,
                        ),
                    )
                    # 原 SESSION 行软删（invalid_at=now）
                    conn.execute(
                        "UPDATE episodes SET invalid_at = ? WHERE rowid = ?",
                        (now_iso, rid),
                    )
                except (ValueError, sqlite3.Error) as exc:
                    logger.warning("consolidate_session_to_user rowid=%s 失败: %s", eid, exc)
            conn.commit()

        async with self.db_lock:
            await asyncio.to_thread(_sync_consolidate)
        logger.debug(
            "sqlite_vector consolidate_session_to_user %s→%s n=%d",
            scope_from,
            scope_to,
            len(rowids),
        )

    async def fetch_episodes_for_consolidation(
        self,
        scope_session: str,
        scope_user: str,
        key: str,
    ) -> list[dict[str, Any]]:
        """读取 scope_session/scope_user 下指定 key 的有效 episode 元数据（巩固批处理专用）。

        返回 list[dict]，每项键：
          episode_id       — str(rowid)
          scope            — scope 字段原值
          key              — key 字段原值
          content          — 内容全文（供 salience 代理解析）
          valid_at         — valid_at 原始 ISO 字符串（调用方自行 fromisoformat）
          access_count     — int（None 时归 0）
          consolidation_count — int（None 时归 0）
          decay_weight     — float（None 时归 1.0）

        只返回 invalid_at IS NULL（有效）行；不含 embedding（避免大对象进记忆层）。
        SQLite I/O 经 asyncio.to_thread 桥接；db_lock 串行化读段（与 apply_decay_weights 对齐）。
        """
        conn = self.conn

        def _sync() -> list[tuple]:
            return conn.execute(
                "SELECT rowid, scope, key, content, valid_at, "
                "access_count, consolidation_count, decay_weight "
                "FROM episodes "
                "WHERE (scope = ? OR scope = ?) AND key = ? AND invalid_at IS NULL",
                (scope_session, scope_user, key),
            ).fetchall()

        async with self.db_lock:
            rows = await asyncio.to_thread(_sync)

        result: list[dict[str, Any]] = []
        for rowid, scope, ep_key, content, valid_at_str, ac, cc, dw in rows:
            result.append(
                {
                    "episode_id": str(rowid),
                    "scope": scope,
                    "key": ep_key,
                    "content": content,
                    "valid_at": valid_at_str,
                    "access_count": ac if ac is not None else 0,
                    "consolidation_count": cc if cc is not None else 0,
                    "decay_weight": dw if dw is not None else 1.0,
                }
            )
        return result

    async def aclose(self) -> None:
        """异步关闭 SQLite 连接（to_thread 桥接，可安全在 async 上下文调用）。"""
        conn = self.conn

        def _sync_close() -> None:
            conn.close()

        async with self.db_lock:
            await asyncio.to_thread(_sync_close)
        logger.debug("sqlite_vector aclose")

    def close(self) -> None:
        """关闭 SQLite 连接（同步便捷方法，供非 async 上下文使用）。

        仅无运行中 event loop 的上下文（teardown / 测试 fixture 同步清理）使用；
        生产异步路径用 aclose()。
        """
        self.conn.close()


def cosine(a: list[float], b: list[float]) -> float:
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


def coerce_dt(value: Any) -> datetime | None:
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


def group_id(scope: str, key: str | None) -> str:
    """构造 Graphiti group_id：`scope[_key]`，并把非 [A-Za-z0-9_-] 字符替换为 `_`。

    Graphiti 的 validate_group_id 只允许字母数字/横杠/下划线——冒号等会报
    GroupIdValidationError，故不能用 `f"{scope}:{key}"`。读写须用同一构造保证可召回。
    """
    raw = scope if key is None else f"{scope}_{key}"
    return "".join(c if (c.isascii() and (c.isalnum() or c in "_-")) else "_" for c in raw)
