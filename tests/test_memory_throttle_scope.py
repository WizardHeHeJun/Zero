"""T6.6 记忆节流与作用域：仅任务完成节点写；每次读写带显式 scope。（G4 / pitfalls 1,2）"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from src.memory.client import MemoryClient
from src.memory.types import Scope
from src.orchestration.runner import run
from src.orchestration.state import Stimulus


class SpyMemory(MemoryClient):
    """记录每次 write 调用，验证节流与显式 scope。"""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[tuple[str, Scope, str]] = []

    async def write(
        self,
        content: str,
        *,
        scope: Scope,
        key: str = "default",
        valid_at: datetime | None = None,
    ) -> None:
        self.writes.append((content, scope, key))
        await super().write(content, scope=scope, key=key, valid_at=valid_at)


async def test_writes_only_at_task_complete_with_explicit_scope() -> None:
    spy = SpyMemory()
    await run(
        [Stimulus(name="win", goal_congruence=0.8, intensity=0.9)],
        thread_id="m1",
        memory=spy,
        rng_seed=1,
    )
    # 一条刺激恰好 2 次写（session 事件 + user 倾向），均在 supervisor 任务完成节点
    assert len(spy.writes) == 2
    assert all(isinstance(w[1], Scope) for w in spy.writes)
    assert {w[1] for w in spy.writes} == {Scope.SESSION, Scope.USER}


def test_write_requires_explicit_scope() -> None:
    mem = MemoryClient()
    with pytest.raises(ValueError):
        asyncio.run(mem.write("x", scope=None))  # type: ignore[arg-type]


async def test_write_count_equals_run_count() -> None:
    """P2-3 节流精确计数：N 次 runner.run 后 memory.write 恰好 N*2 次（每轮 session+user）。

    验证 Supervisor 记忆节流契约（memory-rules #1）：
    - 写入只在任务完成节点发生（非每个中间步骤）。
    - 每轮恰好 2 次（session 事件 + user disposition），不是零次、也不是更多次。
    """
    N = 3
    spy = SpyMemory()
    stimuli = [Stimulus(name=f"s{i}", goal_congruence=0.5, intensity=0.8) for i in range(N)]
    for i, stim in enumerate(stimuli):
        await run(
            [stim],
            thread_id=f"throttle-count-{i}",
            memory=spy,
            rng_seed=i,
        )
    # 每轮 2 次（session 事件 + user disposition），共 N*2 次，不多不少
    assert len(spy.writes) == N * 2, (
        f"期望每轮恰好 2 次 write（session+user），{N} 轮共 {N * 2} 次；"
        f"实际 {len(spy.writes)} 次（触发节流契约失败）"
    )
