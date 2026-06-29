"""运行态持久化：默认 LangGraph 内存 Saver，env 可切 SQLite/Postgres（容器化就绪）。

运行态（V(s)、当前后验、AffectState）经此持久化；长期记忆走图谱后端，
二者分离，不互混（见 memory-rules.md #3）。

后端由 env `ZERO_CHECKPOINT_BACKEND` 选择：`memory`（默认，零依赖）/ `sqlite` / `postgres`。
sqlite/postgres 需 `db` extra（langgraph-checkpoint-sqlite/postgres）；缺驱动时回退 InMemory 并
告警——保证本机零依赖可跑、容器内设 env 即切真后端。

存储层为最底层、对上层无感知：从 checkpoint 恢复的自定义类型白名单由上层（编排层）
通过 allowed_types 传入，本模块不硬编码任何上层类型名。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def build_checkpointer(
    allowed_types: Iterable[tuple[str, ...]] | None = None,
    *,
    backend: str | None = None,
) -> BaseCheckpointSaver:
    """构造运行态 Checkpointer。

    backend 取 env `ZERO_CHECKPOINT_BACKEND`：`memory`（默认）/ `sqlite` / `postgres`。
    allowed_types：白名单从 checkpoint 反序列化的自定义类型 (module, qualname)，由编排层提供——
    `allowed_msgpack_modules` 正是 langgraph 反序列化告警建议的格式，勿改成纯模块名字符串列表。
    sqlite/postgres 缺驱动时回退 InMemory（容器内装 `db` extra 后即生效）。
    """
    serde = (
        JsonPlusSerializer(allowed_msgpack_modules=list(allowed_types))
        if allowed_types is not None
        else None
    )
    choice = (backend or os.getenv("ZERO_CHECKPOINT_BACKEND") or "memory").lower()
    if choice == "sqlite":
        saver = _sqlite_saver(serde)
        if saver is not None:
            return saver
    elif choice == "postgres":
        saver = _postgres_saver(serde)
        if saver is not None:
            return saver
    return InMemorySaver(serde=serde) if serde is not None else InMemorySaver()


def _sqlite_saver(serde: JsonPlusSerializer | None) -> BaseCheckpointSaver | None:
    """SQLite 运行态后端；缺 sqlite saver/aiosqlite 依赖时返回 None 触发回退。

    用 **AsyncSqliteSaver**（非同步 `SqliteSaver`）——本系统全程 `ainvoke`，同步 saver 的
    `aget_tuple` 会抛 `NotImplementedError`。`aiosqlite.connect(path)` 惰性连接、saver 首次
    异步调用时 `await setup()` 懒建表（CREATE TABLE IF NOT EXISTS）。
    ⚠ 须在**运行中的事件循环内**构造（`AsyncSqliteSaver.__init__` 调 `get_running_loop`）——
    本项目均在 asyncio.run 内构造（ConversationSession / runner.run），满足。
    """
    try:
        import aiosqlite  # noqa: F401  AsyncSqliteSaver 的连接依赖（db extra 带）
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        logger.warning(
            "ZERO_CHECKPOINT_BACKEND=sqlite 但缺 sqlite saver/aiosqlite 依赖，回退 InMemory"
        )
        return None
    path = os.getenv("ZERO_CHECKPOINT_DB", "data/checkpoints.sqlite3")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = aiosqlite.connect(path)  # 惰性连接；setup 懒建表，无需在此同步 setup()
    return AsyncSqliteSaver(conn, serde=serde) if serde is not None else AsyncSqliteSaver(conn)


def _postgres_saver(serde: JsonPlusSerializer | None) -> BaseCheckpointSaver | None:
    """Postgres 运行态后端（容器化部署目标）——当前**构造期 fail-fast**，未接线。

    本系统全程 `ainvoke`（异步），需 `AsyncPostgresSaver`；但它要求 **awaited 的连接** + **显式
    `await setup()`**（非懒建表，与 AsyncSqliteSaver 不同），而 `build_checkpointer` 是同步函数、
    且在运行中的事件循环内（无法 `await`）——无法干净地同步构造一个可用的异步 PG saver。
    故此处**清晰报错 fail-fast**，而非旧的同步 `PostgresSaver`（它会在首轮 `ainvoke` 抛 cryptic
    `NotImplementedError`，且若静默回退内存会让"以为有 PG 持久"的部署悄悄丢运行态——更危险）。
    部署 PG 时在**异步入口**接线：`async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
    await saver.setup(); ...`（或把 build_checkpointer/会话构造异步化）。本地请用 sqlite / memory。
    """
    raise NotImplementedError(
        "ZERO_CHECKPOINT_BACKEND=postgres 暂未接线：异步图需 AsyncPostgresSaver（须在异步入口"
        " await 构造 + await setup()），同步 PostgresSaver 会在 ainvoke 崩、静默回退内存又会悄悄"
        "丢运行态。本地请改用 ZERO_CHECKPOINT_BACKEND=sqlite 或 memory；部署 PG 时在异步入口接线。"
    )
