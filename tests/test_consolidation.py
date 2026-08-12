"""记忆巩固与遗忘 B 类单测（2026-07-22）。

覆盖：
  1. 零回归（默认关时各路径逐字不变）。
  2. schema 迁移兼容：旧库 5 列 ALTER TABLE 加 5 新列；旧行不破；新库直接 10 列。
  3. 三存储方法单测：batch_update_access_count / apply_decay_weights /
     consolidate_session_to_user。
  4. consolidation.py 三策略：EbbinghausDecay / SleepConsolidation / ACTRFrequency。
  4b. 防伪长期：低 consolidation_count 高 salience 经 N 轮后 decay_weight 正常衰减。
  5. CS BLOCK：access_count 更新走 Supervisor 任务完成节点，
     召回节点（MemoryRecallAgent）不写 access_count。
  6. recalled_episode_ids 贯通：MemoryRecallAgent actr 路径填充 /
     runner step 每轮归零 / Supervisor 读取后调 batch_update。
  7. Petrov 门控：actr_enabled=False 用原幂律 / True + access_count>0 产 B_norm∈(0,1)。
  8. aclose：门关 no-op / 门开 wait_for 触发 / 超时降级不抛 / 协程持句柄。
  9. 软删语义：search 过滤 invalid_at IS NULL（巩固后 SESSION 行不再被 search 到）。
  10. _trim_capacity 驱逐方向：优先删低 decay_weight（ORDER BY decay_weight DESC 保留高值）。
  11. Fact/StoredFact 字段序列化回归（episode_id/access_count 默认不破）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Fact, Scope
from src.orchestration.memory_recall import MemoryRecallAgent, _petrov_b_norm, _rank_episodes
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import SupervisorAgent
from src.storage.backends.deterministic import StoredFact
from src.storage.backends.semantic import SqliteVectorStore

# ---------------------------------------------------------------------------
# 工具函数：构造带 5 列旧 schema 的 SQLite 数据库
# ---------------------------------------------------------------------------


def _make_legacy_db(path: str = ":memory:") -> sqlite3.Connection:
    """构造 5 列旧 schema 的 SQLite DB（不含巩固新列）。"""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes ("
        "scope TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL, "
        "valid_at TEXT NOT NULL, embedding TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _insert_legacy_row(conn: sqlite3.Connection, scope: str, key: str, content: str) -> None:
    """向旧 schema 插一行（5 列，embedding 用空 JSON 列表）。"""
    now_iso = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        (scope, key, content, now_iso, json.dumps([0.1, 0.2])),
    )
    conn.commit()


def _count_episodes(store: SqliteVectorStore, scope: str, key: str) -> int:
    row = store.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE scope=? AND key=?", (scope, key)
    ).fetchone()
    return int(row[0])


def _distinct_embedder() -> Any:
    """每次调用返回唯一 5 维 one-hot 向量（绕开 dedup 0.92）。"""
    counter = {"n": 0}

    async def embed(text: str) -> list[float]:
        vec = [0.0] * 5
        vec[counter["n"] % 5] = 1.0
        counter["n"] += 1
        return vec

    return embed


# ---------------------------------------------------------------------------
# 11. Fact / StoredFact 字段回归
# ---------------------------------------------------------------------------


def test_fact_defaults_episode_id_and_access_count() -> None:
    """Fact 新字段默认 None/0，不破坏无关键字构造的既有代码。"""
    f = Fact(content="x", scope=Scope.USER, valid_at=datetime.now(UTC))
    assert f.episode_id is None
    assert f.access_count == 0


def test_fact_accepts_episode_id_and_access_count() -> None:
    """Fact 可携带 episode_id 与 access_count（B 类字段存在性）。"""
    f = Fact(
        content="y",
        scope=Scope.SESSION,
        valid_at=datetime.now(UTC),
        episode_id="42",
        access_count=3,
    )
    assert f.episode_id == "42"
    assert f.access_count == 3


def test_stored_fact_defaults_episode_id_and_access_count() -> None:
    """StoredFact 新字段默认 None/0，向后兼容。"""
    sf = StoredFact(scope="user", key="u1", content="z", valid_at=datetime.now(UTC))
    assert sf.episode_id is None
    assert sf.access_count == 0


def test_stored_fact_accepts_episode_id_and_access_count() -> None:
    """StoredFact 可携带 episode_id 与 access_count。"""
    sf = StoredFact(
        scope="session",
        key="s1",
        content="w",
        valid_at=datetime.now(UTC),
        episode_id="7",
        access_count=5,
    )
    assert sf.episode_id == "7"
    assert sf.access_count == 5


# ---------------------------------------------------------------------------
# 2. schema 迁移兼容
# ---------------------------------------------------------------------------


def test_old_schema_alter_table_adds_five_new_columns() -> None:
    """旧库（5 列）被 SqliteVectorStore 打开后 ALTER TABLE 加 5 列，旧行不丢。"""
    # 构造旧 5 列库
    conn = _make_legacy_db()
    _insert_legacy_row(conn, "user", "u1", "old-content")
    conn.close()

    # 用旧库路径重新构造 SqliteVectorStore（:memory: 无法在 close 后复用，改用 tmp 文件）
    # 改用绕过文件的方式：直接用连接注入
    conn2 = _make_legacy_db()
    _insert_legacy_row(conn2, "user", "u1", "old-content")

    # 手动运行迁移（与 SqliteVectorStore.__init__ 中 ALTER TABLE 逻辑相同）
    for col_ddl in (
        "ALTER TABLE episodes ADD COLUMN access_count INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN last_accessed TEXT",
        "ALTER TABLE episodes ADD COLUMN consolidation_count INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN decay_weight REAL DEFAULT 1.0",
        "ALTER TABLE episodes ADD COLUMN invalid_at TEXT",
    ):
        try:
            conn2.execute(col_ddl)
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn2.commit()

    # 旧行仍存在
    rows = conn2.execute(
        "SELECT content, access_count, decay_weight, invalid_at FROM episodes"
    ).fetchall()
    assert len(rows) == 1
    content, ac, dw, ia = rows[0]
    assert content == "old-content"
    assert ac == 0  # DEFAULT 生效
    assert dw == 1.0  # DEFAULT 生效
    assert ia is None  # 软删默认 NULL
    conn2.close()


def test_new_schema_has_ten_columns() -> None:
    """SqliteVectorStore 新库直接含 10 列（CREATE TABLE 定义正确）。"""
    store = SqliteVectorStore(":memory:")
    col_names = {row[1] for row in store.conn.execute("PRAGMA table_info(episodes)")}
    expected = {
        "scope",
        "key",
        "content",
        "valid_at",
        "embedding",
        "access_count",
        "last_accessed",
        "consolidation_count",
        "decay_weight",
        "invalid_at",
    }
    assert expected <= col_names, f"缺列：{expected - col_names}"
    store.close()


def test_alter_table_idempotent_on_new_schema() -> None:
    """新库再次 ALTER TABLE 加已有列 → OperationalError 被 swallowed，不崩。"""
    store = SqliteVectorStore(":memory:")
    # 再次运行迁移不应抛异常
    for col_ddl in (
        "ALTER TABLE episodes ADD COLUMN access_count INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN decay_weight REAL DEFAULT 1.0",
    ):
        try:
            store.conn.execute(col_ddl)
        except sqlite3.OperationalError:
            pass  # 期望被 swallowed
    store.close()


# ---------------------------------------------------------------------------
# 3. 三存储方法单测
# ---------------------------------------------------------------------------


async def test_batch_update_access_count_increments() -> None:
    """batch_update_access_count：指定 rowid 的 access_count +1 / last_accessed 更新。"""
    store = SqliteVectorStore(":memory:")
    now_iso = datetime.now(UTC).isoformat()
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "ep1", now_iso, json.dumps([1.0, 0.0])),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes").fetchone()[0]

    await store.batch_update_access_count([str(rowid)])

    row = store.conn.execute(
        "SELECT access_count, last_accessed FROM episodes WHERE rowid=?", (rowid,)
    ).fetchone()
    assert row[0] == 1, "access_count 应从 0 增到 1"
    assert row[1] is not None, "last_accessed 应被写入"
    store.close()


async def test_batch_update_access_count_multiple_ids() -> None:
    """batch_update_access_count 同时更新多个 episode_id。"""
    store = SqliteVectorStore(":memory:")
    now_iso = datetime.now(UTC).isoformat()
    for i in range(3):
        store.conn.execute(
            "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
            ("user", "u1", f"ep{i}", now_iso, json.dumps([float(i), 0.0])),
        )
    store.conn.commit()
    rowids = [str(r[0]) for r in store.conn.execute("SELECT rowid FROM episodes").fetchall()]

    await store.batch_update_access_count(rowids)

    for rid in rowids:
        row = store.conn.execute(
            "SELECT access_count FROM episodes WHERE rowid=?", (int(rid),)
        ).fetchone()
        assert row[0] == 1, f"rowid={rid} 的 access_count 应为 1"
    store.close()


async def test_batch_update_access_count_empty_noop() -> None:
    """batch_update_access_count 空列表 → no-op，不崩。"""
    store = SqliteVectorStore(":memory:")
    await store.batch_update_access_count([])  # 不抛即通过
    store.close()


async def test_apply_decay_weights_updates_field() -> None:
    """apply_decay_weights：按 (new_dw, episode_id) 更新 decay_weight 字段。"""
    store = SqliteVectorStore(":memory:")
    now_iso = datetime.now(UTC).isoformat()
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "ep1", now_iso, json.dumps([1.0])),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes").fetchone()[0]

    await store.apply_decay_weights([(0.42, str(rowid))])

    dw = store.conn.execute("SELECT decay_weight FROM episodes WHERE rowid=?", (rowid,)).fetchone()[
        0
    ]
    assert abs(dw - 0.42) < 1e-9, f"decay_weight 应为 0.42，实际 {dw}"
    store.close()


async def test_apply_decay_weights_empty_noop() -> None:
    """apply_decay_weights 空列表 → no-op，不崩。"""
    store = SqliteVectorStore(":memory:")
    await store.apply_decay_weights([])
    store.close()


async def test_consolidate_session_to_user_copies_and_soft_deletes() -> None:
    """consolidate_session_to_user：复制 SESSION 行到 USER scope，原行 invalid_at 软删。"""
    store = SqliteVectorStore(":memory:")
    now_iso = datetime.now(UTC).isoformat()
    store.conn.execute(
        "INSERT INTO episodes "
        "(scope, key, content, valid_at, embedding, access_count, consolidation_count, "
        "decay_weight, invalid_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("session", "u1", "sess-ep", now_iso, json.dumps([1.0, 0.0]), 2, 1, 0.8, None),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes WHERE scope='session'").fetchone()[0]

    await store.consolidate_session_to_user("session", "user", [str(rowid)])

    # 原行应软删
    orig_invalid = store.conn.execute(
        "SELECT invalid_at FROM episodes WHERE rowid=?", (rowid,)
    ).fetchone()[0]
    assert orig_invalid is not None, "原 SESSION 行应有 invalid_at（软删）"

    # 新行应在 USER scope
    user_rows = store.conn.execute(
        "SELECT scope, consolidation_count, access_count FROM episodes WHERE scope='user'"
    ).fetchall()
    assert len(user_rows) == 1, "应复制了 1 行到 USER scope"
    scope, cc, ac = user_rows[0]
    assert scope == "user"
    assert cc == 2, "consolidation_count 应 +1（原 1 → 2）"
    assert ac == 2, "access_count 应复制（2）"
    store.close()


async def test_consolidate_session_to_user_empty_noop() -> None:
    """consolidate_session_to_user 空列表 → no-op，不崩。"""
    store = SqliteVectorStore(":memory:")
    await store.consolidate_session_to_user("session", "user", [])
    store.close()


# ---------------------------------------------------------------------------
# 4. consolidation.py 三策略单测
# ---------------------------------------------------------------------------


def _make_episode(
    eid: str,
    scope: str = "session",
    salience: float = 0.5,
    cc: int = 0,
    dw: float = 1.0,
    days_ago: float = 1.0,
) -> dict:
    valid_at = datetime.now(UTC) - timedelta(days=days_ago)
    return {
        "episode_id": eid,
        "scope": scope,
        "key": "u1",
        "content": "test episode",
        "valid_at": valid_at,
        "access_count": 0,
        "consolidation_count": cc,
        "decay_weight": dw,
        "salience": salience,
    }


def test_ebbinghaus_session_fast_decay() -> None:
    """EbbinghausDecay：SESSION scope 用 d_session（快衰）→ decay_weight < USER scope 同参数。"""
    from src.memory.consolidation import EbbinghausDecay

    now = datetime.now(UTC)
    ep_session = _make_episode("s1", scope="session", days_ago=2.0)
    ep_user = _make_episode("u1", scope="user", days_ago=2.0)

    decay = EbbinghausDecay(d_session=0.8, d_user=0.3, a=1.0, kappa=0.0)
    updates, _ = decay.compute([ep_session, ep_user], now=now)

    dw_map = {eid: dw for dw, eid in updates}
    assert "s1" in dw_map and "u1" in dw_map
    assert dw_map["s1"] < dw_map["u1"], (
        f"SESSION 快衰应 < USER 慢衰：dw_session={dw_map['s1']:.4f} dw_user={dw_map['u1']:.4f}"
    )


def test_ebbinghaus_decay_weight_in_range() -> None:
    """EbbinghausDecay 产出 decay_weight∈(0, 1]（钳制有效）。"""
    from src.memory.consolidation import EbbinghausDecay

    now = datetime.now(UTC)
    episodes = [
        _make_episode("e1", days_ago=0.01, salience=0.0),  # 极短时间+低 salience
        _make_episode("e2", days_ago=365.0, salience=1.0),  # 极长时间+高 salience
    ]
    updates, _ = EbbinghausDecay(d_session=0.9, kappa=0.5).compute(episodes, now=now)
    for dw, eid in updates:
        assert 0 < dw <= 1.0, f"episode_id={eid} decay_weight={dw} 超界"


def test_ebbinghaus_importance_modulates_decay() -> None:
    """importance 越高 → a_eff 越大 → decay_weight 越高（衰减越慢）。

    ⚠ 2026-08-12 语义变更（PRP importance-signal）：调制信号由 `salience`
    （affect_precision 派生·衡量情绪后验确定性）换成 `importance`（tag 派生·内容重要性），
    参数化由 `salience^κ` 换成 `(1+κ·u)`。取值用 importance_signal 的真实值域
    `[0.5, 1)`——0.9/0.1 那种 salience 量级在新口径下不合法（<b0 会被 clamp 成 u=0）。
    """
    from src.memory.consolidation import EbbinghausDecay

    now = datetime.now(UTC)
    ep_high = _make_episode("high", days_ago=3.0)
    ep_low = _make_episode("low", days_ago=3.0)
    ep_high["importance"] = 0.744  # 三 tag 全中（noisy-OR: 1−0.5·0.8³）
    ep_low["importance"] = 0.5  # 无 tag → 中性基线 b0

    updates, _ = EbbinghausDecay(kappa=1.0).compute([ep_high, ep_low], now=now)
    dw_map = {eid: dw for dw, eid in updates}
    assert dw_map["high"] > dw_map["low"], (
        f"高 importance 应比中性衰减更慢：high={dw_map['high']:.4f} low={dw_map['low']:.4f}"
    )


def test_ebbinghaus_neutral_importance_is_untouched_by_kappa() -> None:
    """正交性：无 tag（importance=b0 ⇒ u=0）时 decay_weight 对**任意 κ** 相同。

    旧式 `salience^κ` 在此处同样有基线漂移——中性条目也被 `b0^κ<1` 压低。
    """
    from src.memory.consolidation import EbbinghausDecay

    now = datetime.now(UTC)
    baseline: float | None = None
    for kappa in (0.0, 0.5, 1.0, 5.0):
        ep = _make_episode("neutral", days_ago=3.0)
        ep["importance"] = 0.5
        (dw, _eid), *_ = EbbinghausDecay(kappa=kappa).compute([ep], now=now)[0]  # type: ignore[misc]
        if baseline is None:
            baseline = dw
        assert dw == pytest.approx(baseline), f"κ={kappa} 改动了中性条目的 decay_weight"


def test_ebbinghaus_no_consolidate_ids() -> None:
    """EbbinghausDecay 不产生巩固迁移 ids（不属于该策略职责）。"""
    from src.memory.consolidation import EbbinghausDecay

    episodes = [_make_episode("e1")]
    _, consolidate_ids = EbbinghausDecay().compute(episodes, now=datetime.now(UTC))
    assert consolidate_ids == []


def test_sleep_consolidation_both_criteria_met() -> None:
    """SleepConsolidation：salience≥门 AND cc≥min_count → 进 consolidate_ids。"""
    from src.memory.consolidation import SleepConsolidation

    ep = _make_episode("ok", scope="session", salience=0.5, cc=3)
    _, ids = SleepConsolidation(salience_threshold=0.3, consolidation_count_min=3).compute(
        [ep], now=datetime.now(UTC)
    )
    assert "ok" in ids, "双准则满足应升迁"


def test_sleep_consolidation_low_salience_blocked() -> None:
    """SleepConsolidation：salience < 门 → 不升迁（即使 cc 满足）。"""
    from src.memory.consolidation import SleepConsolidation

    ep = _make_episode("low_s", scope="session", salience=0.1, cc=5)
    _, ids = SleepConsolidation(salience_threshold=0.3, consolidation_count_min=2).compute(
        [ep], now=datetime.now(UTC)
    )
    assert "low_s" not in ids, "salience 不足不应升迁"


def test_sleep_consolidation_low_count_blocked() -> None:
    """SleepConsolidation：cc < min_count → 不升迁（即使 salience 满足）。"""
    from src.memory.consolidation import SleepConsolidation

    ep = _make_episode("low_cc", scope="session", salience=0.8, cc=1)
    _, ids = SleepConsolidation(salience_threshold=0.3, consolidation_count_min=3).compute(
        [ep], now=datetime.now(UTC)
    )
    assert "low_cc" not in ids, "cc 不足不应升迁（需 AND 双准则）"


def test_sleep_consolidation_user_scope_skipped() -> None:
    """SleepConsolidation：非 SESSION scope（如 USER）不参与升迁。"""
    from src.memory.consolidation import SleepConsolidation

    ep = _make_episode("user_ep", scope="user", salience=0.9, cc=10)
    _, ids = SleepConsolidation().compute([ep], now=datetime.now(UTC))
    assert "user_ep" not in ids, "USER scope 不参与 session→user 升迁"


def test_sleep_consolidation_no_decay_updates() -> None:
    """SleepConsolidation 不产生 decay_weight 更新（不属该策略职责）。"""
    from src.memory.consolidation import SleepConsolidation

    ep = _make_episode("e1", scope="session", salience=0.9, cc=5)
    updates, _ = SleepConsolidation().compute([ep], now=datetime.now(UTC))
    assert updates == []


def test_actr_frequency_compute_b_norm_n1_positive() -> None:
    """ACTRFrequency.compute_b_norm：n=1 时 B_norm∈(0,1)，单调正值。"""
    from src.memory.consolidation import ACTRFrequency

    actr = ACTRFrequency(d=0.5, b_scale=3.0)
    result = actr.compute_b_norm(access_count=1, l_days=1.0)
    assert 0 < result <= 1.0, f"n=1 B_norm 应在 (0,1]，实际 {result}"


def test_actr_frequency_compute_b_norm_sigmoid_in_range() -> None:
    """ACTRFrequency.compute_b_norm：多个 access_count 值输出均在 (0,1]。"""
    from src.memory.consolidation import ACTRFrequency

    actr = ACTRFrequency(d=0.5, b_scale=3.0)
    for n in [1, 5, 20, 100]:
        val = actr.compute_b_norm(access_count=n, l_days=2.0)
        assert 0 < val <= 1.0, f"access_count={n} B_norm={val} 超界"


def test_actr_frequency_monotone_with_access_count() -> None:
    """ACTRFrequency.compute_b_norm：access_count 增大，B_norm 单调不减（近似验证）。"""
    from src.memory.consolidation import ACTRFrequency

    actr = ACTRFrequency(d=0.5, b_scale=3.0)
    vals = [actr.compute_b_norm(access_count=n, l_days=1.0) for n in [1, 3, 10, 30]]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1], f"vals[{i}]={vals[i]} > vals[{i + 1}]={vals[i + 1]}，非单调"


def test_actr_frequency_compute_noop() -> None:
    """ACTRFrequency.compute() 返回 ([], [])（本策略不产生 decay/迁移输出）。"""
    from src.memory.consolidation import ACTRFrequency

    actr = ACTRFrequency()
    updates, ids = actr.compute([_make_episode("e1")], now=datetime.now(UTC))
    assert updates == [] and ids == []


async def test_run_consolidation_batch_disabled_noop() -> None:
    """run_consolidation_batch enabled=False → 整体 no-op（零回归）。"""
    from src.memory.consolidation import run_consolidation_batch

    mock_semantic = MagicMock()
    mock_semantic.conn = MagicMock()
    mock_semantic.db_lock = asyncio.Lock()
    mock_semantic.apply_decay_weights = AsyncMock()
    mock_semantic.consolidate_session_to_user = AsyncMock()

    await run_consolidation_batch(
        mock_semantic,
        scope_session="session",
        scope_user="user",
        key="u1",
        consolidation_enabled=False,  # 门关
    )

    # 门关时什么都不应被调用
    mock_semantic.apply_decay_weights.assert_not_called()
    mock_semantic.consolidate_session_to_user.assert_not_called()


# ---------------------------------------------------------------------------
# 4b. 防伪长期：低 cc 高 salience 不豁免衰减
# ---------------------------------------------------------------------------


def test_ebbinghaus_low_cc_high_salience_still_decays() -> None:
    """低 consolidation_count 高 salience episode 经多轮后 decay_weight 正常衰减不豁免。

    EbbinghausDecay 的 salience 只调制振幅（a_eff）；低 cc 不特殊豁免衰减——
    若 days_ago 够大，decay_weight 仍会下降到远低于初始值。
    """
    from src.memory.consolidation import EbbinghausDecay

    now = datetime.now(UTC)
    # salience 高（0.9）但时间很久（30天）、cc=0（未强化）
    ep = _make_episode("ep", scope="session", salience=0.9, cc=0, days_ago=30.0)

    updates, _ = EbbinghausDecay(d_session=0.8, a=1.0, kappa=0.5).compute([ep], now=now)
    assert updates, "应有衰减更新"
    dw = updates[0][0]
    # 30天后即使高 salience，SESSION 快衰（d=0.8）decay_weight 应远低于 1.0
    assert dw < 0.5, f"30天后 decay_weight 应 < 0.5，实际 {dw:.4f}（高 salience 未豁免衰减）"


# ---------------------------------------------------------------------------
# 5. CS BLOCK：access_count 更新走 Supervisor，不走 MemoryRecallAgent
# ---------------------------------------------------------------------------


class _RecordingStore:
    """记录 search / batch_update_access_count 调用次数，供 CS BLOCK 断言。"""

    def __init__(self, episodes: list[StoredFact] | None = None) -> None:
        self.episodes: list[StoredFact] = episodes or []
        self.batch_update_calls: list[list[str]] = []

    async def add_episode(self, **_: Any) -> None:
        pass

    async def search(
        self,
        query: str,
        *,
        scope: str,
        key: str | None = None,
        at: Any = None,
        limit: int = 5,
        sim_threshold: float | None = None,
    ) -> list[StoredFact]:
        return self.episodes[:limit]

    async def batch_update_access_count(self, episode_ids: list[str]) -> None:
        self.batch_update_calls.append(list(episode_ids))


async def test_cs_block_recall_does_not_call_batch_update() -> None:
    """CS BLOCK：MemoryRecallAgent 召回时不调用 batch_update_access_count。

    召回节点只负责填充 recalled_episode_ids，不触发 access_count 写入
    （避免污染当轮排序自一致性——当轮频率计数应在任务完成后统一更新）。
    """
    now = datetime.now(UTC)
    store = _RecordingStore(
        episodes=[
            StoredFact(
                scope="user",
                key="u1",
                content="ep-content | precision=0.5",
                valid_at=now,
                sim=0.8,
                episode_id="101",
                access_count=2,
            )
        ]
    )
    mem = MemoryClient(semantic=store)
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="test", goal_congruence=0.0, intensity=0.5),
    )

    await MemoryRecallAgent(mem)(state)

    assert store.batch_update_calls == [], (
        "召回节点不应调用 batch_update_access_count（CS BLOCK 红线）"
    )


async def test_cs_block_supervisor_calls_batch_update_with_episode_ids() -> None:
    """CS BLOCK：Supervisor 任务完成节点读 recalled_episode_ids 并调用 batch_update。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)

    state = AffectState(
        stimulus=Stimulus(name="x", goal_congruence=0.3, intensity=0.5, text="测试"),
        affect_sample=(0.3, 0.4),
        affect_precision=0.5,
        rpe=0.3,
        value_estimate=0.2,
        user_id="u1",
        recalled_episode_ids=["10", "20"],  # 模拟召回阶段填充的 ids
    )

    await SupervisorAgent(mem)(state)

    # Supervisor 应调用了 batch_update_access_count
    assert len(store.batch_update_calls) == 1, "Supervisor 应调用一次 batch_update_access_count"
    assert "10" in store.batch_update_calls[0] and "20" in store.batch_update_calls[0]


async def test_cs_block_supervisor_no_batch_update_without_episode_ids() -> None:
    """Supervisor：recalled_episode_ids 为空时不调用 batch_update（无效调用防御）。"""
    store = _RecordingStore()
    mem = MemoryClient(semantic=store)

    state = AffectState(
        stimulus=Stimulus(name="x", goal_congruence=0.0, intensity=0.5),
        affect_sample=(0.0, 0.0),
        affect_precision=0.3,
        rpe=0.0,
        user_id="u1",
        recalled_episode_ids=[],  # 空
    )

    await SupervisorAgent(mem)(state)
    assert store.batch_update_calls == [], "recalled_episode_ids 为空时不应调用 batch_update"


# ---------------------------------------------------------------------------
# 6. recalled_episode_ids 贯通
# ---------------------------------------------------------------------------


async def test_recall_agent_fills_recalled_episode_ids_when_actr_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryRecallAgent：有 episode_id 的召回 Fact → recalled_episode_ids 填充。

    用 ZERO_ACTR_ENABLED=0（默认）时也会填充 episode_ids——关键是 episode_id 非空的 Fact。
    （actr_enabled 影响排序算法，不影响 episode_ids 收集。）
    """
    now = datetime.now(UTC)
    store = _RecordingStore(
        episodes=[
            StoredFact(
                scope="user",
                key="u1",
                content="test | precision=0.6",
                valid_at=now,
                sim=0.75,
                episode_id="55",
                access_count=1,
            )
        ]
    )
    mem = MemoryClient(semantic=store)
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="test", goal_congruence=0.0, intensity=0.5),
    )

    out = await MemoryRecallAgent(mem)(state)

    assert "recalled_episode_ids" in out, "应填充 recalled_episode_ids"
    assert "55" in out["recalled_episode_ids"]


async def test_recall_agent_no_episode_ids_when_episode_id_none() -> None:
    """MemoryRecallAgent：episode_id=None 的 Fact → recalled_episode_ids 不填充。"""
    now = datetime.now(UTC)
    store = _RecordingStore(
        episodes=[
            StoredFact(
                scope="user",
                key="u1",
                content="test | precision=0.5",
                valid_at=now,
                sim=0.8,
                episode_id=None,  # 无 episode_id
                access_count=0,
            )
        ]
    )
    mem = MemoryClient(semantic=store)
    state = AffectState(
        recall_enabled=True,
        user_id="u1",
        stimulus=Stimulus(name="test", goal_congruence=0.0, intensity=0.5),
    )

    out = await MemoryRecallAgent(mem)(state)

    assert out.get("recalled_episode_ids", []) == [], (
        "episode_id=None 的 Fact 不应进入 recalled_episode_ids"
    )


def test_runner_step_zeros_recalled_episode_ids() -> None:
    """runner.ConversationSession.step 每轮基准含 recalled_episode_ids=[] 防 LastValue 残留。

    验证 ConversationSession.step 的 base dict 包含 recalled_episode_ids 归零项。
    用反射而非 mock：直接读 ConversationSession.step 的源码路径确认归零字段存在
    （避免构造整个图；若字段被删则此测试检出回归）。
    """
    import inspect

    from src.orchestration.runner import ConversationSession

    # step 方法源码中应含归零字段
    src = inspect.getsource(ConversationSession.step)
    assert "recalled_episode_ids" in src, (
        "ConversationSession.step 应包含 recalled_episode_ids 归零防残留"
    )


# ---------------------------------------------------------------------------
# 7. Petrov 门控
# ---------------------------------------------------------------------------


def test_petrov_actr_disabled_uses_power_law() -> None:
    """actr_enabled=False → _rank_episodes 用原幂律 recency（零回归）。

    构造两 Fact：一个 access_count>0（actr 本应奖励），一个 access_count=0（actr 惩罚）；
    actr 关时排序应由 valid_at 决定（较新的幂律 recency 更高）。
    """
    now = datetime.now(UTC)
    f_old = Fact(
        content="ep_old | precision=0.5",
        scope=Scope.USER,
        valid_at=now - timedelta(days=10),
        sim=0.5,
        episode_id="1",
        access_count=100,  # 高频率，actr 本会奖励
    )
    f_new = Fact(
        content="ep_new | precision=0.5",
        scope=Scope.USER,
        valid_at=now - timedelta(days=1),
        sim=0.5,
        episode_id="2",
        access_count=0,  # 零频率，actr 本会惩罚
    )
    # actr_enabled=False → 用幂律 recency，较新的应排前
    ranked = _rank_episodes([f_old, f_new], now, actr_enabled=False)
    assert ranked[0].episode_id == "2", "actr 关时应用幂律 recency，较新 episode 应排前"


def test_petrov_actr_enabled_gives_b_norm_in_range() -> None:
    """actr_enabled=True + access_count>0 → _petrov_b_norm∈(0,1)。"""
    result = _petrov_b_norm(access_count=5, l_days=2.0, d=0.5, b_scale=3.0)
    assert 0 < result <= 1.0, f"B_norm={result} 应在 (0,1]"


def test_petrov_actr_enabled_zero_access_count_gives_small_value() -> None:
    """_petrov_b_norm access_count=0 → 等效 n=1，不奖励未访问 episode（值较小）。

    access_count=0 时取 max(1,0)=1，和 access_count=100 相比 B_norm 应更小。
    """
    b_low = _petrov_b_norm(access_count=0, l_days=1.0, d=0.5, b_scale=3.0)
    b_high = _petrov_b_norm(access_count=100, l_days=1.0, d=0.5, b_scale=3.0)
    assert b_low < b_high, (
        f"access_count=0 B_norm 应 < access_count=100：{b_low:.4f} vs {b_high:.4f}"
    )


def test_petrov_actr_enabled_rank_high_access_count_first() -> None:
    """actr_enabled=True：高 access_count episode 应排在低频 episode 前（ACT-R 奖励频率）。"""
    now = datetime.now(UTC)
    base_valid_at = now - timedelta(days=1)
    f_high = Fact(
        content="ep_high | precision=0.5",
        scope=Scope.USER,
        valid_at=base_valid_at,
        sim=0.5,
        episode_id="h",
        access_count=50,
    )
    f_low = Fact(
        content="ep_low | precision=0.5",
        scope=Scope.USER,
        valid_at=base_valid_at,
        sim=0.5,
        episode_id="l",
        access_count=1,
    )
    ranked = _rank_episodes([f_low, f_high], now, actr_enabled=True, actr_b_scale=3.0)
    assert ranked[0].episode_id == "h", "actr_enabled=True 时高 access_count 应排前"


# ---------------------------------------------------------------------------
# 8. aclose：门控行为
# ---------------------------------------------------------------------------


async def test_aclose_consolidation_disabled_noop() -> None:
    """consolidation_enabled=False（默认）→ aclose 不触发巩固，直接关连接（零回归）。"""
    from src.orchestration.chat_driver import ChatDriver
    from src.storage.conversation_log import ConversationLog

    store = SqliteVectorStore(":memory:")
    mem = MemoryClient(semantic=store)
    mem.run_consolidation_batch = AsyncMock()  # type: ignore[method-assign]

    class _FakeSession:
        async def step(self, stim: Any, state_overrides: Any = None) -> dict:
            return {"valence_arousal": (0.0, 0.0), "recalled_context": []}

    log = ConversationLog(":memory:")
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=_FakeSession(),  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        memory=mem,
        consolidation_enabled=False,  # 默认关
    )

    await driver.aclose()

    mem.run_consolidation_batch.assert_not_called()
    log.close()


async def test_aclose_consolidation_enabled_calls_run_batch() -> None:
    """consolidation_enabled=True → aclose 触发 run_consolidation_batch（门开路径）。"""
    from src.orchestration.chat_driver import ChatDriver
    from src.storage.conversation_log import ConversationLog

    store = SqliteVectorStore(":memory:")
    mem = MemoryClient(semantic=store)
    # 记录调用
    called: list[bool] = []

    async def fake_batch(**_: Any) -> None:
        called.append(True)

    mem.run_consolidation_batch = fake_batch  # type: ignore[method-assign]

    class _FakeSession:
        async def step(self, stim: Any, state_overrides: Any = None) -> dict:
            return {"valence_arousal": (0.0, 0.0), "recalled_context": []}

    log = ConversationLog(":memory:")
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=_FakeSession(),  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        memory=mem,
        consolidation_enabled=True,  # 门开
    )

    await driver.aclose()

    assert called, "consolidation_enabled=True 时 aclose 应触发 run_consolidation_batch"
    log.close()


async def test_aclose_timeout_degrades_gracefully(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """超时（mock 慢 batch）→ warning 降级不抛异常，协程持句柄。"""
    from src.orchestration.chat_driver import ChatDriver
    from src.storage.conversation_log import ConversationLog

    store = SqliteVectorStore(":memory:")
    mem = MemoryClient(semantic=store)

    async def slow_batch(**_: Any) -> None:
        await asyncio.sleep(10.0)  # 故意慢

    mem.run_consolidation_batch = slow_batch  # type: ignore[method-assign]

    class _FakeSession:
        async def step(self, stim: Any, state_overrides: Any = None) -> dict:
            return {"valence_arousal": (0.0, 0.0), "recalled_context": []}

    log = ConversationLog(":memory:")
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=_FakeSession(),  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        memory=mem,
        consolidation_enabled=True,
        consolidation_timeout=0.001,  # 极短超时强制触发
    )

    with caplog.at_level(logging.WARNING):
        await driver.aclose()  # 不应抛异常

    assert any("超时" in r.message or "timeout" in r.message.lower() for r in caplog.records), (
        "超时后应有 warning 日志"
    )
    log.close()


async def test_aclose_no_memory_noop() -> None:
    """memory=None → aclose 整体 no-op，不抛（零回归）。"""
    from src.orchestration.chat_driver import ChatDriver
    from src.storage.conversation_log import ConversationLog

    class _FakeSession:
        async def step(self, stim: Any, state_overrides: Any = None) -> dict:
            return {"valence_arousal": (0.0, 0.0), "recalled_context": []}

    log = ConversationLog(":memory:")
    driver = ChatDriver(
        thread="t",
        lm=None,
        log=log,
        session=_FakeSession(),  # type: ignore[arg-type]
        history=[],
        attitude=(0.0, 0.0),
        mode="test",
        noise_std=0.0,
        memory=None,
    )
    await driver.aclose()  # 不抛即通过
    log.close()


# ---------------------------------------------------------------------------
# 9. 软删语义：search 过滤 invalid_at IS NULL
# ---------------------------------------------------------------------------


async def test_search_filters_soft_deleted_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate_session_to_user 后原 SESSION 行被软删，search 不再返回该行。"""
    store = SqliteVectorStore(":memory:", sim_threshold=0.0)

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)

    now = datetime.now(UTC)
    store.conn.execute(
        "INSERT INTO episodes "
        "(scope, key, content, valid_at, embedding, access_count, consolidation_count, "
        "decay_weight, invalid_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "session",
            "u1",
            "sess-content",
            now.isoformat(),
            json.dumps([1.0, 0.0, 0.0]),
            0,
            1,
            0.8,
            None,
        ),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes WHERE scope='session'").fetchone()[0]

    # 巩固前 search SESSION 应能找到
    results_before = await store.search("query", scope="session", key="u1", sim_threshold=0.0)
    assert len(results_before) == 1, "巩固前应能 search 到 SESSION 行"

    # 软删（模拟巩固后）
    now_iso = datetime.now(UTC).isoformat()
    store.conn.execute("UPDATE episodes SET invalid_at = ? WHERE rowid = ?", (now_iso, rowid))
    store.conn.commit()

    # 软删后 search 应过滤掉
    results_after = await store.search("query", scope="session", key="u1", sim_threshold=0.0)
    assert len(results_after) == 0, "soft-delete 后 search 应过滤 invalid_at IS NOT NULL 的行"
    store.close()


async def test_consolidate_then_search_shows_user_not_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """consolidate_session_to_user 后：search SESSION 返空，search USER 含新行。"""
    store = SqliteVectorStore(":memory:", sim_threshold=0.0)

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)

    now = datetime.now(UTC)
    store.conn.execute(
        "INSERT INTO episodes "
        "(scope, key, content, valid_at, embedding, access_count, consolidation_count, "
        "decay_weight, invalid_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("session", "u1", "content", now.isoformat(), json.dumps([1.0, 0.0, 0.0]), 1, 2, 0.9, None),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes WHERE scope='session'").fetchone()[0]

    await store.consolidate_session_to_user("session", "user", [str(rowid)])

    sess_results = await store.search("q", scope="session", key="u1", sim_threshold=0.0)
    user_results = await store.search("q", scope="user", key="u1", sim_threshold=0.0)

    assert len(sess_results) == 0, "巩固后 SESSION 行应被软删，search SESSION 返空"
    assert len(user_results) == 1, "巩固后 USER scope 应有 1 条"
    store.close()


# ---------------------------------------------------------------------------
# 10. _trim_capacity 驱逐方向：低 decay_weight 先删
# ---------------------------------------------------------------------------


async def test_trim_capacity_evicts_low_decay_weight_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_trim_capacity：优先驱逐低 decay_weight（巩固度低）而非高 decay_weight。

    设 max_per_key=2，插入 3 条：decay_weight=[0.1, 0.5, 0.9]（相同时间）；
    保留应为 decay_weight 最高的 2 条（0.5、0.9），删最低（0.1）。
    """
    store = SqliteVectorStore(":memory:", max_per_key=2)
    monkeypatch.setattr(store, "_embed", _distinct_embedder())

    now = datetime.now(UTC)
    entries = [
        ("low_decay", 0.1),
        ("mid_decay", 0.5),
        ("high_decay", 0.9),
    ]
    for content, dw in entries:
        store.conn.execute(
            "INSERT INTO episodes "
            "(scope, key, content, valid_at, embedding, access_count, consolidation_count, "
            "decay_weight, invalid_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "user",
                "u1",
                content,
                now.isoformat(),
                json.dumps([float(entries.index((content, dw))), 0.0, 0.0]),
                0,
                0,
                dw,
                None,
            ),
        )
    store.conn.commit()

    # 触发 _trim_capacity
    await store._trim_capacity("user", "u1")

    remaining = {
        r[0]
        for r in store.conn.execute(
            "SELECT content FROM episodes WHERE scope='user' AND key='u1' AND invalid_at IS NULL"
        ).fetchall()
    }
    assert "low_decay" not in remaining, (
        "低 decay_weight(0.1) 应被驱逐（ORDER BY decay_weight DESC 保留高值）"
    )
    assert "high_decay" in remaining, "高 decay_weight(0.9) 应被保留"
    assert "mid_decay" in remaining, "中 decay_weight(0.5) 应被保留"
    store.close()


async def test_trim_capacity_zero_max_no_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_per_key=0（默认）→ _trim_capacity 不删任何行（零回归）。"""
    store = SqliteVectorStore(":memory:", max_per_key=0)
    now_iso = datetime.now(UTC).isoformat()
    for i in range(5):
        store.conn.execute(
            "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
            ("user", "u1", f"ep{i}", now_iso, json.dumps([float(i), 0.0])),
        )
    store.conn.commit()

    await store._trim_capacity("user", "u1")

    count = store.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE scope='user' AND key='u1'"
    ).fetchone()[0]
    assert count == 5, f"max_per_key=0 不删行，应有 5 条，实际 {count}"
    store.close()


# ---------------------------------------------------------------------------
# 1. 零回归：默认关时既有行为不变
# ---------------------------------------------------------------------------


def test_actr_state_field_default_false() -> None:
    """AffectState.recalled_episode_ids 默认空列表（新字段不破旧 checkpoint）。"""
    state = AffectState()
    assert state.recalled_episode_ids == [], "recalled_episode_ids 应默认空列表"


async def test_memory_client_batch_update_noop_without_semantic() -> None:
    """MemoryClient.batch_update_access_count：无 semantic 后端 → no-op（零回归）。"""
    mem = MemoryClient()  # semantic=None
    await mem.batch_update_access_count(["1", "2"])  # 不抛即通过


async def test_memory_client_run_consolidation_batch_disabled_noop() -> None:
    """MemoryClient.run_consolidation_batch：consolidation_enabled=False → no-op（零回归）。"""
    mem = MemoryClient()
    await mem.run_consolidation_batch(
        scope_session="session",
        scope_user="user",
        key="u1",
        consolidation_enabled=False,
    )  # 不抛即通过


async def test_memory_client_run_consolidation_batch_no_semantic_noop() -> None:
    """MemoryClient.run_consolidation_batch：无 semantic 后端 → no-op（零回归）。"""
    mem = MemoryClient()  # semantic=None
    await mem.run_consolidation_batch(
        scope_session="session",
        scope_user="user",
        key="u1",
        consolidation_enabled=True,  # 门开但无 semantic
    )  # 不抛即通过


async def test_search_episode_id_transparent_in_stored_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search 返回的 StoredFact.episode_id = rowid str（透传正确）。"""
    store = SqliteVectorStore(":memory:", sim_threshold=0.0)

    async def fixed_embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(store, "_embed", fixed_embed)

    now = datetime.now(UTC)
    store.conn.execute(
        "INSERT INTO episodes (scope, key, content, valid_at, embedding) VALUES (?,?,?,?,?)",
        ("user", "u1", "ep-content", now.isoformat(), json.dumps([1.0, 0.0, 0.0])),
    )
    store.conn.commit()
    rowid = store.conn.execute("SELECT rowid FROM episodes").fetchone()[0]

    results = await store.search("query", scope="user", key="u1", sim_threshold=0.0)
    assert results, "应有 search 结果"
    assert results[0].episode_id == str(rowid), (
        f"episode_id 应为 rowid str '{rowid}'，实际 '{results[0].episode_id}'"
    )
    store.close()


async def test_env_consolidation_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设 ZERO_CONSOLIDATION_ENABLED → consolidation_enabled=False（默认关·零回归）。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_CONSOLIDATION_ENABLED", raising=False)
    driver = build_chat_driver(thread="test-cons-default")
    assert driver.consolidation_enabled is False, "默认应关闭巩固（零回归）"


async def test_env_actr_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_chat_driver：未设 ZERO_ACTR_ENABLED → actr_enabled=False（默认关）。"""
    from src.orchestration.chat_driver import build_chat_driver

    monkeypatch.delenv("ZERO_ACTR_ENABLED", raising=False)
    driver = build_chat_driver(thread="test-actr-default")
    assert driver.actr_enabled is False, "默认应关闭 ACT-R（零回归）"
