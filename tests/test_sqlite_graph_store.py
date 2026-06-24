"""SqliteGraphStore（本地落盘长期记忆）单测：时序失效、持久化、工厂选后端。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.storage.graph_store import (
    InMemoryGraphStore,
    SqliteGraphStore,
    build_graph_store,
)


def test_temporal_invalidation_latest_active() -> None:
    store = SqliteGraphStore(":memory:")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    store.add_fact("user", "u1", "mood=neg", t0)
    store.add_fact("user", "u1", "mood=pos", t1)
    active = store.query_facts("user", key="u1")
    assert len(active) == 1
    assert active[0].content == "mood=pos"


def test_query_at_past_time_returns_old_fact() -> None:
    store = SqliteGraphStore(":memory:")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    store.add_fact("user", "u1", "old", t0)
    store.add_fact("user", "u1", "new", t1)
    # 在 t0 与 t1 之间查询 → 仍是 old 有效（时序语境）
    facts = store.query_facts("user", key="u1", at=t0 + timedelta(minutes=30))
    assert len(facts) == 1
    assert facts[0].content == "old"


def test_persistence_across_instances(tmp_path) -> None:
    db = str(tmp_path / "graph.sqlite3")
    s1 = SqliteGraphStore(db)
    s1.add_fact("session", "s1", "event=loss", datetime(2026, 1, 1, tzinfo=UTC))
    # 新实例打开同一文件 → 事实仍在（落盘持久化）
    s2 = SqliteGraphStore(db)
    facts = s2.query_facts("session", key="s1")
    assert len(facts) == 1
    assert facts[0].content == "event=loss"


def test_factory_default_is_memory(monkeypatch) -> None:
    monkeypatch.delenv("ZERO_MEMORY_BACKEND", raising=False)
    assert isinstance(build_graph_store(), InMemoryGraphStore)


def test_factory_sqlite_selected_by_env(monkeypatch) -> None:
    monkeypatch.setenv("ZERO_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("ZERO_GRAPH_DB", ":memory:")
    assert isinstance(build_graph_store(), SqliteGraphStore)
