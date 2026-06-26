"""对话 transcript + 跨重启 attitude 的本地 SQLite 存储（stdlib，无额外依赖）。

原定义在**临时入口** `main.py` 内；迁出到存储层，使「删除 main.py 做主入口迁移」后，
对话历史 / 态度持久化能力不随入口消亡、可被任意入口复用。存储边界见类 docstring。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

DEFAULT_CHAT_HISTORY_PATH = "data/chat_history.sqlite3"


class ConversationLog:
    """对话 transcript 落本地 SQLite：逐轮存、重启重载近 N 轮 → 长期记忆。

    存储边界（与 supervisor 并行、职责不重叠）：
    - 本类管对话运行态：transcript turns 表（每轮 user/assistant 消息）+
      跨重启 attitude meta 表（持久化的慢变情绪基线）。
    - 情感事件/长期 episode/disposition 属长期情感记忆，由 SupervisorAgent 经
      MemoryClient 写入（user/session scope，见 supervisor.py）。
    - 两套存储并行运行——不在此处调用 MemoryClient，不在 supervisor 读写 transcript。
    - 本类仅在交互对话（REPL）路径使用；图内 StateGraph 路径走
      SupervisorAgent→MemoryClient，两条路径互斥，attitude 不双写。
    """

    def __init__(self, path: str = DEFAULT_CHAT_HISTORY_PATH) -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(thread TEXT NOT NULL, ts TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta "
            "(thread TEXT PRIMARY KEY, feeling_v REAL, feeling_a REAL)"
        )
        self.conn.commit()

    def append(self, thread: str, role: str, content: str) -> None:
        """追加一轮消息（user/assistant）到 transcript。"""
        self.conn.execute(
            "INSERT INTO turns (thread, ts, role, content) VALUES (?, ?, ?, ?)",
            (thread, datetime.now(UTC).isoformat(), role, content),
        )
        self.conn.commit()

    def recent(self, thread: str, limit: int = 20) -> list[dict[str, str]]:
        """取该 thread 最近 limit 条，按时间正序返回 [{role, content}…]（喂 LLM 对话历史）。"""
        rows = self.conn.execute(
            "SELECT role, content FROM turns WHERE thread = ? ORDER BY rowid DESC LIMIT ?",
            (thread, limit),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def save_feeling(self, thread: str, feeling: tuple[float, float]) -> None:
        """落盘累积情绪（跨重启续上「情绪积累」，不每次从平静重来）。"""
        self.conn.execute(
            "INSERT INTO meta (thread, feeling_v, feeling_a) VALUES (?, ?, ?) "
            "ON CONFLICT(thread) DO UPDATE SET feeling_v=excluded.feeling_v, "
            "feeling_a=excluded.feeling_a",
            (thread, feeling[0], feeling[1]),
        )
        self.conn.commit()

    def load_feeling(self, thread: str) -> tuple[float, float]:
        """读回累积情绪；无记录则 (0, 0)（平静起步）。"""
        row = self.conn.execute(
            "SELECT feeling_v, feeling_a FROM meta WHERE thread = ?", (thread,)
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else (0.0, 0.0)

    def close(self) -> None:
        """关闭底层 SQLite 连接（测试清理 / 显式释放文件句柄用）。"""
        self.conn.close()
