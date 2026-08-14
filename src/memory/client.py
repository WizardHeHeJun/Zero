"""记忆层读写 API：显式 scope、任务完成节流、封装图谱后端。

上层（编排层/Agent）只能经本 API 访问长期记忆，不得直连图谱/向量库。
写入仅应在 Supervisor 判定「任务完成」的节点调用（节流）；不要在每个
Worker 每步回调里写（见 memory-rules.md #1、pitfalls）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.memory.types import Fact, Scope
from src.memory.utils import importance_signal, parse_importance_tags
from src.storage.graph_store import GraphStore, InMemoryGraphStore, SemanticStore

logger = logging.getLogger(__name__)


class MemoryClient:
    """长期记忆读写入口；对下依赖 GraphStore 协议（非具体类），对上屏蔽存储细节。

    可选注入 `semantic`（SemanticStore，如 Graphiti）：与确定性 GraphStore 并存的
    语义记忆侧信道，承载富 episode 写入（`write_episode`）+ 语义召回（`recall`）。
    未注入时二者 no-op / 返回空——严格零回归，不影响确定性 write/query 路径。
    """

    def __init__(
        self, store: GraphStore | None = None, *, semantic: SemanticStore | None = None
    ) -> None:
        self.store: GraphStore = store if store is not None else InMemoryGraphStore()
        self.semantic: SemanticStore | None = semantic

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
        # 后端失败隔离：底层 GraphStore（如 Neo4j 连不上服务）抛错时降级——只告警不崩主管线。
        # 与语义侧信道 write_episode/recall 同款韧性；scope 校验（编程错误）仍前置抛出、不在此吞。
        try:
            self.store.add_fact(scope=scope.value, key=key, content=content, valid_at=when)
            logger.info("memory.write scope=%s key=%s", scope.value, key)
        except Exception as exc:
            logger.warning(
                "memory.write failed scope=%s key=%s: %s", scope.value, key, exc, exc_info=True
            )

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
        # 后端失败隔离：读失败（如 Neo4j 连不上）降级返回 []（recalled_disposition 退化为 None、
        # appraisal 不偏置），不崩主管线。与 write 同款韧性。
        try:
            stored = self.store.query_facts(scope=scope.value, key=key, at=at)
        except Exception as exc:
            logger.warning(
                "memory.query failed scope=%s key=%s: %s", scope.value, key, exc, exc_info=True
            )
            return []
        return [
            Fact(content=s.content, scope=scope, valid_at=s.valid_at, key=s.key) for s in stored
        ]

    async def write_episode(
        self,
        content: str,
        *,
        scope: Scope,
        key: str = "default",
        valid_at: datetime | None = None,
        embed_text: str | None = None,
    ) -> None:
        """写一条自然语言 episode 到语义记忆（Graphiti 抽取实体/关系入图）。

        必须显式 scope；仅应在任务完成节点调用（节流，同 write）。无语义后端时 no-op
        （严格零回归）。与确定性 `write` 互补：write 存结构化标量事实，write_episode 存
        供语义召回的富文本。`embed_text`（可选）= 用于检索的语义 gist，与存储全文 `content`
        分离以免元数据稀释向量（议会 A 召回桥根治）；缺省对全文嵌入（旧行为）。
        """
        if not isinstance(scope, Scope):
            raise ValueError("memory.write_episode 必须显式指定 Scope，禁止默认作用域")
        if self.semantic is None:
            return
        when = valid_at if valid_at is not None else datetime.now(UTC)
        # importance_tag（PRP sleep-consolidation-verdict 案 i·2026-08-13 裁定）：纯 tag 分量
        # 的写入时身份重要性，供 SqliteVectorStore._trim_capacity 驱逐排序第二键消费；
        # 记忆层合法 import 同层 utils（守三层单向）。禁 fold-in 全量 I（含 precision 分量，
        # 会把排序侧刚切干净的「precision 污染重要性判断」缝回驱逐侧，见 design.md §三议题3）。
        importance_tag = importance_signal(parse_importance_tags(content))
        try:
            await self.semantic.add_episode(
                scope=scope.value,
                key=key,
                content=content,
                valid_at=when,
                embed_text=embed_text,
                importance_tag=importance_tag,
            )
            logger.info("memory.write_episode scope=%s key=%s", scope.value, key)
        except Exception as exc:
            logger.warning(
                "write_episode failed scope=%s key=%s: %s", scope.value, key, exc, exc_info=True
            )
            return

    async def aclose(self) -> None:
        """关闭底层语义后端连接（duck-typing，异步优先）。

        检测顺序：优先 aclose（SqliteVectorStore）→ 回退 close 且为 coroutinefunction
        （async def close，如 GraphitiGraphStore）→ 均无则 no-op。
        确保两种后端（SqliteVectorStore/GraphitiGraphStore）都能被正确关闭。
        """
        import inspect

        if self.semantic is None:
            return
        if hasattr(self.semantic, "aclose"):
            await self.semantic.aclose()  # type: ignore[union-attr]
            logger.debug("memory_client aclose delegated to semantic.aclose")
        elif hasattr(self.semantic, "close") and inspect.iscoroutinefunction(
            self.semantic.close  # type: ignore[union-attr]
        ):
            await self.semantic.close()  # type: ignore[union-attr]
            logger.debug("memory_client aclose delegated to semantic.close (async)")

    async def recall(
        self,
        query: str,
        *,
        scope: Scope,
        key: str | None = None,
        at: datetime | None = None,
        limit: int = 5,
    ) -> list[Fact]:
        """语义召回（向量/图检索），按作用域与时间语境返回相关事实。

        必须显式 scope。无语义后端时返回 `[]`（零回归）——与确定性 `query`（结构化
        (scope,key) 命中）不同，本方法吃 query 文本做语义检索。
        episode_id / access_count 从 StoredFact 透传（semantic 路径填真值；
        Graphiti 等无此字段的后端经 getattr 安全降级为 None/0）。
        """
        if not isinstance(scope, Scope):
            raise ValueError("memory.recall 必须显式指定 Scope，禁止默认作用域")
        if self.semantic is None:
            return []
        try:
            stored = await self.semantic.search(
                query, scope=scope.value, key=key, at=at, limit=limit
            )
            logger.debug(
                "memory.recall scope=%s key=%s q=%s n=%d", scope.value, key, query, len(stored)
            )
            return [
                Fact(
                    content=s.content,
                    scope=scope,
                    valid_at=s.valid_at,
                    key=s.key,
                    sim=s.sim,
                    episode_id=getattr(s, "episode_id", None),
                    access_count=getattr(s, "access_count", 0),
                )
                for s in stored
            ]
        except Exception as exc:
            logger.warning(
                "recall failed scope=%s key=%s: %s", scope.value, key, exc, exc_info=True
            )
            return []

    async def batch_update_access_count(self, episode_ids: list[str]) -> None:
        """节流更新 episode access_count（ACT-R 频率；仅 Supervisor 任务完成节点调用）。

        经 duck-type 检测 semantic 是否具备 batch_update_access_count 能力；
        无语义后端或后端不支持时 no-op（严格零回归）。
        不在召回节点调用（CS BLOCK：召回时更新会污染当轮排序自一致性）。
        层归属修正：Supervisor 通过本方法访问 semantic，不直连存储层（守三层单向）。
        """
        if not episode_ids or self.semantic is None:
            return
        if not hasattr(self.semantic, "batch_update_access_count"):
            return
        try:
            await self.semantic.batch_update_access_count(episode_ids)  # type: ignore[union-attr]
            logger.debug("memory_client batch_update_access_count n=%d", len(episode_ids))
        except Exception as exc:
            logger.warning(
                "batch_update_access_count failed n=%d: %s", len(episode_ids), exc, exc_info=True
            )

    async def run_consolidation_batch(
        self,
        *,
        scope_session: str,
        scope_user: str,
        key: str,
        consolidation_enabled: bool = False,
        d_session: float = 0.8,
        d_user: float = 0.3,
        actr_b_scale: float = 3.0,
        tag_importance_enabled: bool = False,
    ) -> None:
        """会话结束时触发记忆衰减批处理（经 MemoryClient 封装·守三层单向）。

        把 self.semantic 传给 consolidation.run_consolidation_batch；
        ChatDriver 调本方法而非直接传 semantic（层归属修正）。
        consolidation_enabled=False（默认）→ no-op（严格零回归）。
        无语义后端时同样 no-op。
        ACT-R 频率归一仅控召回侧 recency 替换，巩固批处理不消费，故无 actr_enabled 参数。
        """
        if not consolidation_enabled or self.semantic is None:
            return
        # 延迟 import：consolidation 属记忆层，aclose 路径才触发，不在热路径
        from src.memory.consolidation import run_consolidation_batch

        try:
            await run_consolidation_batch(
                self.semantic,
                scope_session=scope_session,
                scope_user=scope_user,
                key=key,
                consolidation_enabled=consolidation_enabled,
                d_session=d_session,
                d_user=d_user,
                actr_b_scale=actr_b_scale,
                tag_importance_enabled=tag_importance_enabled,
            )
        except Exception as exc:
            logger.warning("run_consolidation_batch failed key=%s: %s", key, exc, exc_info=True)
