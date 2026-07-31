"""身份自陈旁路的端到端归因验证（PRP identity-fact-write-bypass · V4/V5/V7）。

跑在**真实 salience 分布**上：台词与逐轮 (precision, rpe) 均固化自 2026-07-30 的
100 轮实跑（`fixtures_hundred_turn_script`），不是测试自己编的常数。

⚠ 采用**归因式双臂**而非「查关键词在不在库」：后者双向无判别力——
多写一条假身份记忆不会红；旁路完全坏掉也可能因主门自己写了那轮而"通过"。
"""

from __future__ import annotations

import pytest

from src.memory.client import MemoryClient
from src.orchestration.state import AffectState, Stimulus
from src.orchestration.supervisor import IDENTITY_MEMORY_PRECISION, SupervisorAgent
from tests.fixtures_hundred_turn_script import (
    HUNDRED_TURN_SCRIPT,
    IDENTITY_TURNS,
    TURN_SALIENCE_INPUTS,
)


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def add_episode(
        self,
        content: str,
        *,
        scope: str,
        key: str,
        valid_at: object = None,
        embed_text: str | None = None,
    ) -> None:
        self.calls.append({"content": content, "scope": scope, "key": key})

    async def search(self, *args: object, **kwargs: object) -> list:
        return []


def _state(text: str, precision: float, rpe: float) -> AffectState:
    return AffectState(
        stimulus=Stimulus(name=text[:40], text=text, goal_congruence=0.0, intensity=0.0),
        affect_sample=(0.0, 0.0),
        affect_precision=precision,
        rpe=rpe,
        user_id="chat",
    )


async def _run_arm(
    *, bypass_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[set[int], _RecordingStore]:
    """跑完整 100 轮，返回 (发生写入的轮次集合, store)。"""
    monkeypatch.setenv("ZERO_IDENTITY_FACT_BYPASS", "1" if bypass_enabled else "0")
    store = _RecordingStore()
    agent = SupervisorAgent(MemoryClient(semantic=store))
    written: set[int] = set()
    for turn, (text, (precision, rpe)) in enumerate(
        zip(HUNDRED_TURN_SCRIPT, TURN_SALIENCE_INPUTS, strict=True), start=1
    ):
        before = len(store.calls)
        await agent(_state(text, precision, rpe))
        if len(store.calls) > before:
            written.add(turn)
    return written, store


async def test_fixture_alignment() -> None:
    """台词与 salience 输入必须一一对应，否则下面所有断言都在测错的东西。"""
    assert len(HUNDRED_TURN_SCRIPT) == len(TURN_SALIENCE_INPUTS) == 100


async def test_dual_arm_diff_is_exactly_identity_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """V4 归因式双臂：开/关旁路的写入轮次差集**恰好**等于身份轮集合。

    多一条即红（旁路收进了不该收的），少一条即红（旁路没起作用）。
    """
    off, _ = await _run_arm(bypass_enabled=False, monkeypatch=monkeypatch)
    on, _ = await _run_arm(bypass_enabled=True, monkeypatch=monkeypatch)

    assert on - off == set(IDENTITY_TURNS), (
        f"旁路新增的写入轮次 {sorted(on - off)} != 期望 {sorted(IDENTITY_TURNS)}"
    )
    assert off - on == set(), f"旁路不应让原本会写的轮次不写：{sorted(off - on)}"
    # 防「本用例没在测东西」：关闭臂必须确实写了不少轮，否则差集恒等于 on
    assert len(off) > 20, f"关闭臂只写了 {len(off)} 轮，salience 分布可能没生效"


async def test_identity_turn_is_missed_by_main_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """前提复现：身份轮在关闭旁路时**确实**被主门丢弃——这是本特性存在的理由。

    若这条变绿失败（即主门自己就写了身份轮），说明前提已不成立，
    上面的双臂差集也就不再证明旁路有效，须重新评估本特性。
    """
    off, _ = await _run_arm(bypass_enabled=False, monkeypatch=monkeypatch)
    for turn in IDENTITY_TURNS:
        assert turn not in off, f"轮 {turn} 在关闭旁路时也被写入，本特性的前提已不成立"


async def test_identity_episode_carries_recall_clearing_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V7 的确定性部分：身份 episode 的 precision 须使 Hill 归一高于召回注入门。

    真实语义召回排名需要 embedding、属实机 smoke；此处钉住可确定性验证的那一半——
    若 precision 不过门，该 episode 永远进不了 LLM 注意力预算，写了也白写。
    """
    _, store = await _run_arm(bypass_enabled=True, monkeypatch=monkeypatch)
    identity_episodes = [c for c in store.calls if "我叫林川" in c["content"]]
    assert identity_episodes, "身份 episode 未写入"
    assert f"precision={IDENTITY_MEMORY_PRECISION:.2f}" in identity_episodes[0]["content"]

    inject_min = 0.5  # ZERO_RECALL_INJECT_MIN 默认值
    scale = 30.0  # ZERO_RECALL_IMPORTANCE_SCALE 默认值
    normalized = IDENTITY_MEMORY_PRECISION / (IDENTITY_MEMORY_PRECISION + scale)
    assert normalized > inject_min, (
        f"归一 importance {normalized:.3f} 未过注入门 {inject_min}，写进去也召不回"
    )
    # 对照：若沿用该轮真实 precision(8.56)，归一后 0.222 < 0.5 —— 正是首版遗漏的缺口
    raw = TURN_SALIENCE_INPUTS[3][0]
    assert raw / (raw + scale) < inject_min, "该对照失效说明注入门前提变了，须重新评估架构决策 F"
