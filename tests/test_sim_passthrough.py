"""D4：语义检索 sim 透传链路单测（PR-1）。

覆盖：
  - StoredFact / Fact 新增 sim 字段默认 0.0（零回归：旧构造不破坏）。
  - SqliteVectorStore.search 把余弦相似度填进 StoredFact.sim。
  - MemoryClient.recall 把 StoredFact.sim 透传到 Fact.sim。
  - 确定性 query 路径 Fact.sim 恒 0.0（不走语义检索，无 sim 概念）。

全部 monkeypatch _embed，不调真 embedding/网络（无 openai 依赖）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.storage.backends.deterministic import StoredFact
from src.storage.backends.semantic import SqliteVectorStore


def test_stored_fact_sim_defaults_zero() -> None:
    """StoredFact 不传 sim 时默认 0.0（零回归：现有所有构造处无需改动）。"""
    fact = StoredFact(scope="user", key="u1", content="c", valid_at=datetime.now(UTC))
    assert fact.sim == 0.0


def test_fact_sim_defaults_zero() -> None:
    """Fact 不传 sim 时默认 0.0。"""
    from src.memory.types import Fact

    fact = Fact(content="c", scope=Scope.USER, valid_at=datetime.now(UTC))
    assert fact.sim == 0.0


def _insert_episode(store: SqliteVectorStore, content: str, embedding: list[float]) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", content, t0.isoformat(), json.dumps(embedding)),
    )
    store.conn.commit()


async def test_sqlite_search_fills_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    """search 结果的 StoredFact.sim == query 与 episode 的余弦相似度。"""
    store = SqliteVectorStore(":memory:")
    _insert_episode(store, "aligned", [1.0, 0.0, 0.0])  # 与 query 同向 → cos=1.0

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)

    results = await store.search("q", scope="user", key="u1", sim_threshold=0.0)
    assert results, "应有命中"
    assert results[0].content == "aligned"
    assert abs(results[0].sim - 1.0) < 1e-9, f"sim 应为 1.0，实际 {results[0].sim}"


async def test_memory_client_recall_propagates_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    """MemoryClient.recall 返回的 Fact.sim 等于底层 StoredFact.sim。"""
    store = SqliteVectorStore(":memory:")
    _insert_episode(store, "aligned", [1.0, 0.0, 0.0])

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)
    mem = MemoryClient(semantic=store)

    facts = await mem.recall("q", scope=Scope.USER, key="u1", limit=5)
    assert facts, "应有召回"
    assert abs(facts[0].sim - 1.0) < 1e-9, f"Fact.sim 应透传 1.0，实际 {facts[0].sim}"


async def test_query_path_sim_zero() -> None:
    """确定性 query 路径（非语义检索）Fact.sim 恒 0.0（零回归）。"""
    mem = MemoryClient()  # 默认 InMemoryGraphStore
    await mem.write("disposition stimulus=x value=-0.3", scope=Scope.USER, key="u1")
    facts = await mem.query("disposition", scope=Scope.USER, key="u1")
    assert facts
    assert all(f.sim == 0.0 for f in facts), "确定性 query 无语义相似度，sim 应恒 0.0"
