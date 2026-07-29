"""会话 registry：`session_id → ConversationSession` 的异步生命周期管理。

zero-link 边界是**有状态 actor**（值表跨轮累积、可选 mood/皮质醇三时间尺度），
故工具面必须携带会话身份（`open/step/close`）——本 registry 守 server 侧的会话表。
client 无状态：`session_id` 由 client 持有、每轮随 `zero.step` 透传（见边界回执 §一）。

并发安全：`asyncio.Lock` 保护表读写（同一 server 事件循环内多工具并发调用）。
"""

from __future__ import annotations

import asyncio

from src.orchestration.runner import ConversationSession


class SessionRegistry:
    """守 `session_id → ConversationSession` 的会话表（异步、锁保护）。

    表级 `self.lock` 只保护表结构读写（open/get/close）；每会话另配一把 `asyncio.Lock`
    串行化对**同一** thread_id 的 `step`——LangGraph checkpointer 的读-改-写非原子，
    并发 `ainvoke(同 thread_id)` 会竞态（http 传输下 client 可能并发；stdio 顺序则天然无碍）。
    """

    def __init__(self) -> None:
        self.sessions: dict[str, ConversationSession] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.lock = asyncio.Lock()

    async def open(self, session_id: str, session: ConversationSession) -> None:
        """登记一个新会话 + 其专属 step 锁（同 id 覆写，调用方保证 id 唯一：uuid4）。"""
        async with self.lock:
            self.sessions[session_id] = session
            self.locks[session_id] = asyncio.Lock()

    async def get(self, session_id: str) -> ConversationSession | None:
        """取会话；未知 id 返回 None（调用方据此抛结构化工具错误）。"""
        async with self.lock:
            return self.sessions.get(session_id)

    async def acquire(
        self, session_id: str
    ) -> tuple[ConversationSession | None, asyncio.Lock | None]:
        """原子取出会话 + 其 step 锁（未知 id → (None, None)）。

        调用方 `async with lock:` 后再 `await session.step(...)`，串行化同会话的并发 step。
        锁获取在表锁下原子完成，无 get/lock 之间的 TOCTOU。
        """
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None, None
            return session, self.locks[session_id]

    async def close(self, session_id: str) -> bool:
        """释放会话 + 其锁；返回是否确有其会话（未知 id 幂等返回 False，不报错）。

        本方法只负责从表中弹出引用；会话自身的运行态连接（aiosqlite 等）由调用方在此之前
        `await session.aclose()` 关闭（见 `server.close_session`），别依赖对象回收。
        """
        async with self.lock:
            self.locks.pop(session_id, None)
            return self.sessions.pop(session_id, None) is not None

    async def count(self) -> int:
        """当前活跃会话数（观测用）。"""
        async with self.lock:
            return len(self.sessions)
