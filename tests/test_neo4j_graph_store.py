"""Neo4jGraphStore（长期记忆 Neo4j 后端）测试：工厂回退（本机可跑）+ 时序失效（需实例）。

工厂回退用例不依赖驱动/实例，本机直跑；时序语义用例 importorskip + 连接探测，
无可用 Neo4j 时优雅 skip，有实例（如 docker compose 起的 neo4j）时真跑。
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from src.storage import graph_store as gs
from src.storage.graph_store import InMemoryGraphStore, build_graph_store


def test_factory_neo4j_falls_back_without_driver(monkeypatch) -> None:
    """选 neo4j 但驱动不可用（_neo4j_store 返回 None）→ 回退 InMemory，不抛。"""
    monkeypatch.setattr(gs, "_neo4j_store", lambda: None)
    monkeypatch.setenv("ZERO_MEMORY_BACKEND", "neo4j")
    assert isinstance(build_graph_store(), InMemoryGraphStore)


def test_neo4j_store_helper_returns_none_when_import_fails(monkeypatch) -> None:
    """缺 neo4j 包时 _neo4j_store 探测失败返回 None（触发工厂回退）。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "neo4j":
            raise ImportError("simulated missing neo4j")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gs._neo4j_store() is None


def _live_store() -> gs.Neo4jGraphStore:
    """连真实 Neo4j；缺驱动或无实例时 skip，并清库保证用例隔离。"""
    pytest.importorskip("neo4j")
    uri = os.getenv("ZERO_NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("ZERO_NEO4J_USER", "neo4j")
    password = os.getenv("ZERO_NEO4J_PASSWORD", "password")
    store = gs.Neo4jGraphStore(uri, user, password)
    try:
        store.driver.verify_connectivity()
    except Exception:
        pytest.skip("无可用 Neo4j 实例，跳过集成用例")
    with store.driver.session() as session:
        session.run("MATCH (f:Fact) DETACH DELETE f")
    return store


def test_temporal_invalidation_latest_active() -> None:
    store = _live_store()
    try:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = t0 + timedelta(hours=1)
        store.add_fact("user", "u1", "mood=neg", t0)
        store.add_fact("user", "u1", "mood=pos", t1)
        active = store.query_facts("user", key="u1")
        assert len(active) == 1
        assert active[0].content == "mood=pos"
    finally:
        store.close()


def test_query_at_past_time_returns_old_fact() -> None:
    store = _live_store()
    try:
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = t0 + timedelta(hours=1)
        store.add_fact("user", "u1", "old", t0)
        store.add_fact("user", "u1", "new", t1)
        facts = store.query_facts("user", key="u1", at=t0 + timedelta(minutes=30))
        assert len(facts) == 1
        assert facts[0].content == "old"
    finally:
        store.close()
