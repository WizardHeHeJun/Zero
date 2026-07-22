"""记忆层公共类型：作用域与事实。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Scope(StrEnum):
    """记忆作用域；读写**必须显式**指定，禁止默认（见 memory-rules.md #2）。"""

    USER = "user"  # 跨会话长期（如长期情绪倾向/人格）
    SESSION = "session"  # 单会话（如当前情绪事件）
    GROUP = "group"  # 多 Agent 共享


@dataclass
class Fact:
    """一条带时间维度的记忆事实。"""

    content: str
    scope: Scope
    valid_at: datetime
    key: str = "default"
    sim: float = 0.0  # D4：语义召回的余弦相似度（透传自 StoredFact.sim；确定性 query 路径为 0.0）
    # 记忆巩固与遗忘字段（B 类·默认零回归·不加 pydantic Field 约束防 checkpoint 反序列化失败）
    episode_id: str | None = None  # 对应 SqliteVectorStore 内 rowid（str），无语义后端时 None
    access_count: int = 0  # ACT-R 频率累计（Petrov 2006 B 近似）；Supervisor 节流更新
