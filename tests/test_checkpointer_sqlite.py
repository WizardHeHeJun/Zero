"""sqlite 运行态后端 async 回归：checkpointer=sqlite 必须支持 ainvoke。

复现并锁定真后端 dogfood 暴露的 bug——`_sqlite_saver` 曾用同步 `SqliteSaver`，而系统全程
`ainvoke`，`aget_tuple` 抛 `NotImplementedError: SqliteSaver does not support async methods`。
修法：改用 `AsyncSqliteSaver`（aiosqlite 惰性连接 + 懒建表）。

缺 db extra（langgraph-checkpoint-sqlite/aiosqlite）时 importorskip 跳过（CI 轻量环境）。
"""

from __future__ import annotations

import pytest

from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.storage.checkpointer import build_checkpointer


async def test_sqlite_checkpointer_supports_ainvoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """checkpointer=sqlite 下整图 ainvoke 一轮不抛 NotImplementedError（异步 saver 真生效）。"""
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    pytest.importorskip("aiosqlite")

    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite3"))

    from src.memory.client import MemoryClient
    from src.orchestration.graph import build_graph
    from src.orchestration.state import Stimulus

    # build_checkpointer 在本异步测试内调用 → 有运行中的事件循环（AsyncSqliteSaver.__init__ 需要）
    saver = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    graph = build_graph(saver, MemoryClient())

    out = await graph.ainvoke(
        {"stimulus": Stimulus(name="hi", goal_congruence=0.2, intensity=0.5), "rng_seed": 0},
        config={"configurable": {"thread_id": "t-sqlite-async"}},
    )
    assert out is not None  # 跑到此处即未抛 NotImplementedError（旧同步 saver 会在此崩）


async def test_sqlite_checkpointer_persists_across_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """同一 thread 跨两次 build（模拟重启）能从 sqlite 落盘恢复运行态（value_table 续上）。"""
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    pytest.importorskip("aiosqlite")

    db = str(tmp_path / "ckpt.sqlite3")
    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("ZERO_CHECKPOINT_DB", db)

    from src.memory.client import MemoryClient
    from src.orchestration.graph import build_graph
    from src.orchestration.state import Stimulus

    cfg = {"configurable": {"thread_id": "t-persist"}}
    stim = Stimulus(name="loss", goal_congruence=-0.6, intensity=0.8)

    graph1 = build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())
    await graph1.ainvoke({"stimulus": stim, "rng_seed": 0}, config=cfg)

    # 新建一套 saver（同 db 文件）模拟重启，应能读回上一轮 checkpoint
    graph2 = build_graph(build_checkpointer(ALLOWED_CHECKPOINT_TYPES), MemoryClient())
    state = await graph2.aget_state(cfg)
    assert state is not None and state.values, "应从 sqlite 落盘恢复到上一轮运行态"
