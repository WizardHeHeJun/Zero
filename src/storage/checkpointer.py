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
import sqlite3
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
    """SQLite 运行态后端（本地落盘）；缺 langgraph-checkpoint-sqlite 时返回 None 触发回退。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning(
            "ZERO_CHECKPOINT_BACKEND=sqlite 但缺 langgraph-checkpoint-sqlite，回退 InMemory"
        )
        return None
    path = os.getenv("ZERO_CHECKPOINT_DB", "data/checkpoints.sqlite3")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=serde) if serde is not None else SqliteSaver(conn)


def _postgres_saver(serde: JsonPlusSerializer | None) -> BaseCheckpointSaver | None:
    """Postgres 运行态后端（容器化部署目标）；缺 langgraph-checkpoint-postgres 时返回 None。

    DSN 取 env `ZERO_PG_DSN`。具体构造按容器内 langgraph 版本接线（首次需 `.setup()`）；
    本机无驱动、不参与单测，于容器内验证。
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        logger.warning(
            "ZERO_CHECKPOINT_BACKEND=postgres 但缺 langgraph-checkpoint-postgres，回退 InMemory"
        )
        return None
    dsn = os.getenv("ZERO_PG_DSN", "postgresql://postgres:postgres@localhost:5432/zero")
    saver = PostgresSaver.from_conn_string(dsn)
    saver.setup()
    return saver
