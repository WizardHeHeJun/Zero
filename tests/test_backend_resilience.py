"""后端切换韧性回归：

- Fix 1：ZERO_CHECKPOINT_BACKEND=postgres 构造期 fail-fast（清晰 NotImplementedError），
  不再用同步 PostgresSaver 在首轮 ainvoke 抛 cryptic 错、也不静默回退内存丢运行态。
- Fix 2：确定性记忆 write/query 对底层 GraphStore 失败（如 Neo4j 连不上）失败隔离——
  降级（write 跳过 / query 返 []）不崩主管线；但 scope 校验（编程错误）仍前置抛出、不被吞。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.runner import ALLOWED_CHECKPOINT_TYPES
from src.storage.checkpointer import build_checkpointer

# --------------------------------------------------------------------------- #
# Fix 1：postgres fail-fast
# --------------------------------------------------------------------------- #


def test_postgres_backend_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_CHECKPOINT_BACKEND=postgres → build_checkpointer 抛清晰 NotImplementedError。"""
    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "postgres")
    with pytest.raises(NotImplementedError, match="postgres"):
        build_checkpointer(ALLOWED_CHECKPOINT_TYPES)


def test_memory_backend_still_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """memory 后端正常构造（确认 fail-fast 只针对 postgres，没误伤默认路径）。"""
    monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "memory")
    saver = build_checkpointer(ALLOWED_CHECKPOINT_TYPES)
    assert saver is not None


# --------------------------------------------------------------------------- #
# Fix 2：确定性记忆后端失败隔离
# --------------------------------------------------------------------------- #


class _FaultyStore:
    """模拟连不上的后端（如未起服务的 Neo4j）：所有读写抛错。"""

    def add_fact(self, **kwargs: object) -> None:
        raise RuntimeError("backend down (simulated)")

    def query_facts(self, **kwargs: object) -> list:
        raise RuntimeError("backend down (simulated)")


async def test_write_isolates_backend_failure(caplog: pytest.LogCaptureFixture) -> None:
    """store.add_fact 抛错 → write 不向上抛（降级 + warning），主管线不崩。"""
    mem = MemoryClient(store=_FaultyStore())  # type: ignore[arg-type]
    import logging

    with caplog.at_level(logging.WARNING):
        await mem.write("内容", scope=Scope.USER, key="u1")  # 不抛即通过
    assert any("memory.write failed" in r.message for r in caplog.records)


async def test_query_isolates_backend_failure() -> None:
    """store.query_facts 抛错 → query 降级返回 []（recalled_disposition 退化为 None）。"""
    mem = MemoryClient(store=_FaultyStore())  # type: ignore[arg-type]
    result = await mem.query("q", scope=Scope.USER, key="u1")
    assert result == []


async def test_scope_validation_still_raises_not_swallowed() -> None:
    """失败隔离不得吞掉 scope 校验——非 Scope 仍前置抛 ValueError（编程错误，非后端故障）。"""
    mem = MemoryClient(store=_FaultyStore())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Scope"):
        await mem.write("x", scope="user", key="u1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Scope"):
        await mem.query("q", scope="user", key="u1")  # type: ignore[arg-type]


async def test_healthy_store_unaffected() -> None:
    """正常后端（默认 InMemoryGraphStore）write→query 往返不受隔离逻辑影响。"""
    mem = MemoryClient()
    await mem.write("disposition stimulus=x value=-0.3", scope=Scope.USER, key="u1")
    facts = await mem.query("disposition", scope=Scope.USER, key="u1")
    assert facts and "value=-0.3" in facts[0].content
    assert isinstance(facts[0].valid_at, datetime)
