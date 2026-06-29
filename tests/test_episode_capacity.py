"""D7：episode 容量上限剪裁单测（PR-2）。

覆盖：
  - ZERO_EPISODE_MAX_PER_KEY=0（默认）→ 不剪裁，写 N 条仍 N 条（零回归）。
  - ZERO_EPISODE_MAX_PER_KEY=3 → 写 5 条只留按 valid_at 最新 3 条，删最旧 2 条。
  - 剪裁只在 add_episode（写路径）发生；search（读路径）不触发删除（守 BLOCK-C）。
  - 剪裁按 (scope,key) 隔离，不影响其他 key。

全部 monkeypatch _embed（distinct 向量避开 dedup），不调真 embedding/网络。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.storage.backends.semantic import SqliteVectorStore


def _distinct_embedder() -> object:
    """返回一个每次调用产出唯一 5 维 one-hot 向量的 fake _embed（避开 dedup 0.92）。"""
    counter = {"n": 0}

    async def embed(text: str) -> list[float]:
        vec = [0.0] * 5
        vec[counter["n"] % 5] = 1.0
        counter["n"] += 1
        return vec

    return embed


def _count(store: SqliteVectorStore, scope: str, key: str) -> int:
    row = store.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE scope=? AND key=?", (scope, key)
    ).fetchone()
    return int(row[0])


async def test_capacity_zero_no_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认 ZERO_EPISODE_MAX_PER_KEY=0 → 写 5 条全保留（零回归）。"""
    monkeypatch.delenv("ZERO_EPISODE_MAX_PER_KEY", raising=False)
    store = SqliteVectorStore(":memory:")
    monkeypatch.setattr(store, "_embed", _distinct_embedder())
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        await store.add_episode(
            scope="user", key="u1", content=f"ep{i}", valid_at=t0 + timedelta(days=i)
        )
    assert _count(store, "user", "u1") == 5


async def test_capacity_limits_keep_newest(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_EPISODE_MAX_PER_KEY=3 → 写 5 条只留按 valid_at 最新 3 条（ep2/ep3/ep4）。"""
    monkeypatch.setenv("ZERO_EPISODE_MAX_PER_KEY", "3")
    store = SqliteVectorStore(":memory:")
    monkeypatch.setattr(store, "_embed", _distinct_embedder())
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        await store.add_episode(
            scope="user", key="u1", content=f"ep{i}", valid_at=t0 + timedelta(days=i)
        )
    assert _count(store, "user", "u1") == 3
    rows = store.conn.execute(
        "SELECT content FROM episodes WHERE scope='user' AND key='u1'"
    ).fetchall()
    kept = {r[0] for r in rows}
    assert kept == {"ep2", "ep3", "ep4"}, f"应保留最新 3 条，实际 {kept}"


async def test_capacity_isolated_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """剪裁按 (scope,key) 隔离：u1 超量被剪不影响 u2。"""
    monkeypatch.setenv("ZERO_EPISODE_MAX_PER_KEY", "2")
    store = SqliteVectorStore(":memory:")
    monkeypatch.setattr(store, "_embed", _distinct_embedder())
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(4):
        await store.add_episode(
            scope="user", key="u1", content=f"a{i}", valid_at=t0 + timedelta(days=i)
        )
    await store.add_episode(scope="user", key="u2", content="b0", valid_at=t0)
    assert _count(store, "user", "u1") == 2
    assert _count(store, "user", "u2") == 1


async def test_search_does_not_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    """search（读路径）不触发剪裁删除（守 BLOCK-C：读节点无写副作用）。"""
    monkeypatch.setenv("ZERO_EPISODE_MAX_PER_KEY", "3")
    store = SqliteVectorStore(":memory:")
    monkeypatch.setattr(store, "_embed", _distinct_embedder())
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        await store.add_episode(
            scope="user", key="u1", content=f"ep{i}", valid_at=t0 + timedelta(days=i)
        )
    before = _count(store, "user", "u1")
    assert before == 3  # add 时已剪到 3

    # search 若误触发剪裁会让 _trim_capacity 崩
    store._trim_capacity = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("search 不应触发 _trim_capacity")
    )
    await store.search("q", scope="user", key="u1", sim_threshold=0.0)
    assert _count(store, "user", "u1") == before, "search 不应改变 episode 数量"
